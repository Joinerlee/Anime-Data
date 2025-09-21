# Spring ↔ FastAPI 애니메이션 추천 시스템 연동 튜토리얼

## 📋 목차
1. [시스템 개요](#시스템-개요)
2. [Spring 측 구현 가이드](#spring-측-구현-가이드)
3. [API 연동 방법](#api-연동-방법)
4. [데이터 동기화](#데이터-동기화)
5. [에러 처리](#에러-처리)
6. [성능 최적화](#성능-최적화)

---

## 🏗️ 시스템 개요

### 역할 분담
```
┌─────────────────┐    ┌──────────────────────┐
│   Spring Boot   │    │   FastAPI Service   │
│   (메인 서버)    │    │   (추천 & 검색)      │
├─────────────────┤    ├──────────────────────┤
│ • 사용자 관리   │    │ • 개인화 추천        │
│ • 인증/인가     │◄──►│ • 의미론적 검색      │
│ • 평점 관리     │    │ • 임베딩 처리        │
│ • UI/비즈니스   │    │ • pgvector 검색      │
└─────────────────┘    └──────────────────────┘
```

### 데이터 동기화 패턴
- **애니메이션 데이터**: Spring → FastAPI (플래그 기반)
- **사용자 취향**: Spring → FastAPI (일일 12시 자동)
- **추천 결과**: FastAPI → Spring (실시간 콜백)

---

## 🔧 Spring 측 구현 가이드

### 1. 기본 설정

#### application.yml
```yaml
# FastAPI 서비스 설정
recommendation:
  service:
    url: http://localhost:8000
    api-key: ${RECOMMENDATION_API_KEY:your-secret-key}
    timeout: 30s

# 스케줄러 설정
scheduler:
  user-sync:
    cron: "0 0 12 * * ?" # 매일 12시
    enabled: true
```

#### 의존성 (build.gradle)
```gradle
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
    implementation 'org.springframework.boot:spring-boot-starter-webflux' // WebClient용
    implementation 'org.springframework.boot:spring-boot-starter-validation'
}
```

### 2. 추천 서비스 클라이언트 구현

#### RecommendationServiceClient.java
```java
@Component
@Slf4j
public class RecommendationServiceClient {

    private final WebClient webClient;

    @Value("${recommendation.service.url}")
    private String baseUrl;

    @Value("${recommendation.service.api-key}")
    private String apiKey;

    public RecommendationServiceClient(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder
            .defaultHeader("Authorization", "Bearer " + apiKey)
            .defaultHeader("Content-Type", "application/json")
            .build();
    }

    /**
     * 개인화 추천 요청
     */
    public Mono<RecommendationResponse> getRecommendations(RecommendationRequest request) {
        return webClient.post()
            .uri(baseUrl + "/api/recommend")
            .bodyValue(request)
            .retrieve()
            .onStatus(HttpStatus::isError, response -> {
                log.error("추천 요청 실패: {}", response.statusCode());
                return Mono.error(new RecommendationException("추천 서비스 오류"));
            })
            .bodyToMono(RecommendationResponse.class)
            .timeout(Duration.ofSeconds(30));
    }

    /**
     * 의미론적 검색
     */
    public Mono<SearchResponse> searchAnimations(String query, int limit) {
        SearchRequest request = SearchRequest.builder()
            .query(query)
            .limit(limit)
            .build();

        return webClient.post()
            .uri(baseUrl + "/api/search/semantic")
            .bodyValue(request)
            .retrieve()
            .bodyToMono(SearchResponse.class);
    }

    /**
     * 신규 애니메이션 동기화 플래그
     */
    public Mono<Void> syncNewAnimations(List<Long> animationIds) {
        AnimationSyncRequest request = AnimationSyncRequest.builder()
            .type("new_animations")
            .animationIds(animationIds)
            .build();

        return webClient.post()
            .uri(baseUrl + "/api/internal/animations/sync")
            .bodyValue(request)
            .retrieve()
            .toBodilessEntity()
            .then();
    }

    /**
     * 사용자 취향 데이터 동기화
     */
    public Mono<Void> syncUserPreferences(List<UserPreferenceDto> preferences) {
        UserSyncRequest request = UserSyncRequest.builder()
            .userPreferences(preferences)
            .build();

        return webClient.post()
            .uri(baseUrl + "/api/internal/users/sync")
            .bodyValue(request)
            .retrieve()
            .toBodilessEntity()
            .then();
    }
}
```

### 3. 데이터 모델 정의

#### RecommendationRequest.java
```java
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RecommendationRequest {

    @NotBlank
    private String userId;

    @NotNull
    private List<Long> likedAnimeIds;

    private List<Long> dislikedAnimeIds = new ArrayList<>();

    @Min(1) @Max(50)
    private int nRecommendations = 10;
}
```

#### RecommendationResponse.java
```java
@Data
@NoArgsConstructor
@AllArgsConstructor
public class RecommendationResponse {

    private List<RecommendationItem> recommendations;
    private String method;
    private long responseTimeMs;

    @Data
    public static class RecommendationItem {
        private Long id;
        private String title;
        private double finalScore;
        private double contentScore;
        private double collaborativeScore;
        private double genreSimilarity;
        private double preferenceScore;
        private List<String> animeGenres;
        private List<String> userTopGenres;
        private List<String> matchedGenres;
        private String recommendationReason;
        private String recommendationMethod;
        private String genres;
        private Integer year;
        private String synopsis;
    }
}
```

### 4. 서비스 레이어 구현

#### AnimationRecommendationService.java
```java
@Service
@Slf4j
@Transactional(readOnly = true)
public class AnimationRecommendationService {

    private final RecommendationServiceClient recommendationClient;
    private final UserPreferenceRepository userPreferenceRepository;
    private final AnimationRepository animationRepository;

    /**
     * 사용자 맞춤 추천
     */
    public Mono<List<AnimationDto>> getPersonalizedRecommendations(String userId, int count) {
        // 1. 사용자 선호도 조회
        UserPreference userPref = userPreferenceRepository.findByUserId(userId)
            .orElseThrow(() -> new UserNotFoundException("사용자를 찾을 수 없습니다: " + userId));

        // 2. 추천 요청 생성
        RecommendationRequest request = RecommendationRequest.builder()
            .userId(userId)
            .likedAnimeIds(userPref.getLikedAnimeIds())
            .dislikedAnimeIds(userPref.getDislikedAnimeIds())
            .nRecommendations(count)
            .build();

        // 3. 추천 서비스 호출
        return recommendationClient.getRecommendations(request)
            .map(response -> response.getRecommendations().stream()
                .map(this::convertToDto)
                .collect(Collectors.toList()));
    }

    /**
     * 검색 기능
     */
    public Mono<List<AnimationDto>> searchAnimations(String query, int limit) {
        return recommendationClient.searchAnimations(query, limit)
            .map(response -> response.getResults().stream()
                .map(this::convertToDto)
                .collect(Collectors.toList()));
    }

    private AnimationDto convertToDto(RecommendationResponse.RecommendationItem item) {
        return AnimationDto.builder()
            .id(item.getId())
            .title(item.getTitle())
            .genres(Arrays.asList(item.getGenres().split("\\|")))
            .year(item.getYear())
            .synopsis(item.getSynopsis())
            .recommendationScore(item.getFinalScore())
            .recommendationReason(item.getRecommendationReason())
            .build();
    }
}
```

### 5. 컨트롤러 구현

#### AnimationController.java
```java
@RestController
@RequestMapping("/api/animations")
@Slf4j
public class AnimationController {

    private final AnimationRecommendationService recommendationService;

    /**
     * 개인화 추천
     */
    @GetMapping("/recommendations")
    public Mono<ResponseEntity<ApiResponse<List<AnimationDto>>>> getRecommendations(
            @RequestParam String userId,
            @RequestParam(defaultValue = "10") int count) {

        return recommendationService.getPersonalizedRecommendations(userId, count)
            .map(recommendations -> ResponseEntity.ok(
                ApiResponse.success("추천 조회 성공", recommendations)))
            .onErrorReturn(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(ApiResponse.error("추천 조회 실패")));
    }

    /**
     * 애니메이션 검색
     */
    @GetMapping("/search")
    public Mono<ResponseEntity<ApiResponse<List<AnimationDto>>>> searchAnimations(
            @RequestParam String q,
            @RequestParam(defaultValue = "10") int limit) {

        return recommendationService.searchAnimations(q, limit)
            .map(results -> ResponseEntity.ok(
                ApiResponse.success("검색 성공", results)))
            .onErrorReturn(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(ApiResponse.error("검색 실패")));
    }
}
```

---

## 🔄 데이터 동기화

### 1. 신규 애니메이션 동기화

#### AnimationEventHandler.java
```java
@Component
@Slf4j
public class AnimationEventHandler {

    private final RecommendationServiceClient recommendationClient;

    /**
     * 신규 애니메이션 등록 시 추천 서비스에 동기화
     */
    @EventListener
    @Async
    public void handleNewAnimationEvent(AnimationCreatedEvent event) {
        log.info("신규 애니메이션 동기화: {}", event.getAnimationIds());

        recommendationClient.syncNewAnimations(event.getAnimationIds())
            .doOnSuccess(v -> log.info("애니메이션 동기화 완료"))
            .doOnError(e -> log.error("애니메이션 동기화 실패", e))
            .subscribe();
    }
}
```

### 2. 사용자 취향 일괄 동기화

#### UserPreferenceSyncScheduler.java
```java
@Component
@Slf4j
public class UserPreferenceSyncScheduler {

    private final RecommendationServiceClient recommendationClient;
    private final UserPreferenceRepository userPreferenceRepository;

    /**
     * 매일 12시 사용자 취향 데이터 동기화
     */
    @Scheduled(cron = "0 0 12 * * ?")
    @Transactional(readOnly = true)
    public void syncAllUserPreferences() {
        log.info("일일 사용자 취향 동기화 시작");

        try {
            // 어제 이후 업데이트된 사용자들
            LocalDateTime since = LocalDateTime.now().minusDays(1);
            List<UserPreference> updatedUsers =
                userPreferenceRepository.findUpdatedSince(since);

            if (updatedUsers.isEmpty()) {
                log.info("동기화할 사용자 데이터가 없습니다");
                return;
            }

            // DTO 변환
            List<UserPreferenceDto> preferencesDtos = updatedUsers.stream()
                .map(this::convertToDto)
                .collect(Collectors.toList());

            // 추천 서비스에 동기화
            recommendationClient.syncUserPreferences(preferencesDtos)
                .doOnSuccess(v -> log.info("사용자 취향 동기화 완료: {}명", preferencesDtos.size()))
                .doOnError(e -> log.error("사용자 취향 동기화 실패", e))
                .block(); // 스케줄 작업이므로 동기 처리

        } catch (Exception e) {
            log.error("사용자 취향 동기화 중 오류 발생", e);
        }
    }

    private UserPreferenceDto convertToDto(UserPreference pref) {
        return UserPreferenceDto.builder()
            .userId(pref.getUserId())
            .likedAnimeIds(pref.getLikedAnimeIds())
            .dislikedAnimeIds(pref.getDislikedAnimeIds())
            .build();
    }
}
```

---

## ⚠️ 에러 처리

### 1. 예외 정의

#### RecommendationException.java
```java
public class RecommendationException extends RuntimeException {
    public RecommendationException(String message) {
        super(message);
    }

    public RecommendationException(String message, Throwable cause) {
        super(message, cause);
    }
}
```

### 2. 글로벌 에러 핸들러

#### GlobalExceptionHandler.java
```java
@RestControllerAdvice
@Slf4j
public class GlobalExceptionHandler {

    @ExceptionHandler(RecommendationException.class)
    public ResponseEntity<ApiResponse<Void>> handleRecommendationException(RecommendationException e) {
        log.error("추천 서비스 오류: {}", e.getMessage());
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
            .body(ApiResponse.error("추천 서비스를 일시적으로 사용할 수 없습니다"));
    }

    @ExceptionHandler(WebClientException.class)
    public ResponseEntity<ApiResponse<Void>> handleWebClientException(WebClientException e) {
        log.error("외부 서비스 통신 오류: {}", e.getMessage());
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
            .body(ApiResponse.error("외부 서비스와의 통신에 실패했습니다"));
    }
}
```

### 3. 서킷 브레이커 (선택사항)

#### RecommendationServiceClient.java (개선된 버전)
```java
@Component
public class RecommendationServiceClient {

    @CircuitBreaker(name = "recommendation-service", fallbackMethod = "getFallbackRecommendations")
    @TimeLimiter(name = "recommendation-service")
    @Retry(name = "recommendation-service")
    public Mono<RecommendationResponse> getRecommendations(RecommendationRequest request) {
        // 기존 구현...
    }

    /**
     * 추천 서비스 장애시 폴백
     */
    public Mono<RecommendationResponse> getFallbackRecommendations(RecommendationRequest request, Exception ex) {
        log.warn("추천 서비스 폴백 실행: {}", ex.getMessage());

        // 기본 추천 로직 (예: 인기 애니메이션)
        return Mono.just(RecommendationResponse.builder()
            .recommendations(getPopularAnimations())
            .method("fallback")
            .build());
    }
}
```

---

## 🚀 성능 최적화

### 1. 캐싱 전략

#### CacheConfig.java
```java
@Configuration
@EnableCaching
public class CacheConfig {

    @Bean
    public CacheManager cacheManager() {
        RedisCacheManager.Builder builder = RedisCacheManager
            .RedisCacheManagerBuilder
            .fromConnectionFactory(redisConnectionFactory())
            .cacheDefaults(cacheConfiguration());

        return builder.build();
    }

    private RedisCacheConfiguration cacheConfiguration() {
        return RedisCacheConfiguration.defaultCacheConfig()
            .entryTtl(Duration.ofMinutes(30)) // 추천 결과 30분 캐시
            .serializeKeysWith(RedisSerializationContext.SerializationPair
                .fromSerializer(new StringRedisSerializer()))
            .serializeValuesWith(RedisSerializationContext.SerializationPair
                .fromSerializer(new GenericJackson2JsonRedisSerializer()));
    }
}
```

#### CachedRecommendationService.java
```java
@Service
public class CachedRecommendationService {

    @Cacheable(value = "recommendations", key = "#userId + '_' + #count")
    public Mono<List<AnimationDto>> getPersonalizedRecommendations(String userId, int count) {
        // 기존 추천 로직...
    }

    @CacheEvict(value = "recommendations", key = "#userId + '*'")
    public void evictUserRecommendations(String userId) {
        log.info("사용자 추천 캐시 삭제: {}", userId);
    }
}
```

### 2. 비동기 처리

#### AsyncConfig.java
```java
@Configuration
@EnableAsync
public class AsyncConfig {

    @Bean(name = "recommendationTaskExecutor")
    public TaskExecutor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(5);
        executor.setMaxPoolSize(20);
        executor.setQueueCapacity(100);
        executor.setThreadNamePrefix("Recommendation-");
        executor.initialize();
        return executor;
    }
}
```

---

## 📝 체크리스트

### 초기 설정
- [ ] FastAPI 서비스 URL 및 API 키 설정
- [ ] WebClient 및 의존성 추가
- [ ] 에러 핸들링 구현
- [ ] 캐싱 설정 (선택사항)

### 기본 연동
- [ ] 추천 API 연동
- [ ] 검색 API 연동
- [ ] 사용자 취향 데이터 동기화
- [ ] 신규 애니메이션 플래그 구현

### 운영 준비
- [ ] 모니터링 및 로깅
- [ ] 서킷 브레이커 설정 (선택사항)
- [ ] 성능 테스트
- [ ] 장애 대응 시나리오

---

## 🆘 문제 해결

### 자주 발생하는 문제들

**Q: 추천 서비스에 연결할 수 없어요**
```
A: 1. FastAPI 서비스가 실행 중인지 확인
   2. 방화벽/네트워크 설정 확인
   3. API 키가 올바른지 확인
```

**Q: 추천 결과가 나오지 않아요**
```
A: 1. 사용자에게 충분한 평점 데이터가 있는지 확인
   2. 애니메이션 데이터가 동기화되었는지 확인
   3. PostgreSQL의 pgvector 확장이 설치되었는지 확인
```

**Q: 성능이 느려요**
```
A: 1. Redis 캐싱 활성화
   2. 데이터베이스 인덱스 확인
   3. 비동기 처리 적용
   4. 요청 횟수 제한 (Rate Limiting)
```

---

**🎯 더 자세한 정보가 필요하시면 FastAPI 서비스의 `/docs` 엔드포인트를 확인하세요!**