# main.py
# -----------------------------------------------------------
# AniLIFE 작품 "작품 정보(tab=info)" 크롤러 (Playwright 필수)
# - ID/URL/범위/목록 파일을 받아 INFO 탭만 긁어서 CSV 저장
# - JS 렌더 완료 뒤 섹션(h2/h3 라벨) 기준으로 안전하게 추출
# - 네비게이션은 networkidle 금지 → domcontentloaded/load + 요소 대기
# - [OK]/[SKIP] 로깅, 디버그 HTML/스크린샷 옵션
# -----------------------------------------------------------
import re
import csv
import time
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

CSV_HEADERS = [
    "컨텐츠ID", "URL",
    "제목", "분기", "장르", "테마", "공식 줄거리",
    "성우1", "성우2", "성우3", "성우4", "성우5", "성우6",
    "감독", "애니메이션 제작", "제작"
]

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

TYPE_RE = r"(TV|극장판|영화|ONA|OVA|WEB|Web|스페셜|SP|특별편)"
ROLE_WORDS = ("감독", "애니메이션 제작", "제작")
PLACEHOLDER_SYNOP = ("등록된 줄거리가 없습니다", "줄거리 없음")


# -------------------- 공통 유틸 --------------------
def ensure_info_url(url: str) -> str:
    return url if "tab=info" in url else (url + ("&" if "?" in url else "?") + "tab=info")


def build_info_url(cid: int) -> str:
    return f"https://anilife.app/content/{cid}?tab=info"


# -------------------- JS 파서 --------------------
JS_EXTRACT = r"""
() => {
  // ---- DOM 헬퍼 ----
  const norm = s => (s||"").replace(/\s+/g," ").trim();
  const text = el => el ? norm(el.textContent) : "";
  const by = (sel, root=document) => Array.from(root.querySelectorAll(sel));
  const dedup = arr => Array.from(new Set(arr.filter(Boolean)));
  const hasText = (el, kw) => norm(el?.textContent||"").includes(kw);

  const sectionByH2 = (label) => {
    // h2/h3 텍스트에 label이 포함된 헤더를 찾고, 가장 가까운 <section>을 반환
    const hs = by('h2,h3');
    const h = hs.find(x => hasText(x, label));
    if (!h) return null;
    return h.closest('section') || h.parentElement || h;
  };

  // ---- 결과 객체 ----
  const res = {};

  // ---- 제목 ----
  res.title = text(document.querySelector('h1') || document.querySelector('header h1'));

  // ---- 분기: season 링크 + 타입 텍스트 (백업: 본문에서 정규식) ----
  (() => {
    let seasonTxt = "";
    let typeTxt = "";

    const seasonLink = document.querySelector('a[href^="/season/"]');
    if (seasonLink) {
      seasonTxt = text(seasonLink);
      // 같은 헤더/부모 컨테이너에서 타입 후보(span 등)
      const scope = seasonLink.closest('header, .hero, .banner, ._header, main, section') || document;
      const tCand = by('span, b, strong', scope).map(text).find(t => /^(TV|극장판|영화|ONA|OVA|WEB|Web|스페셜|SP|특별편)$/.test(t));
      typeTxt = tCand || "";
    }

    if (!seasonTxt) {
      const body = text(document.body);
      const m = body.match(/(\d{4})년도\s*([1-4])분기/);
      if (m) {
        const mt = body.match(/(TV|극장판|영화|ONA|OVA|WEB|Web|스페셜|SP|특별편)/);
        seasonTxt = `${m[1]}년도 ${m[2]}분기`;
        typeTxt = mt ? (mt[1]==='WEB'?'Web':mt[1]) : "";
      }
    }

    res.season = seasonTxt + (typeTxt ? ` · ${typeTxt}` : "");
  })();

  // ---- 장르 ----
  (() => {
    const sec = sectionByH2('작품 장르');
    let genres = [];
    if (sec) {
      // 칩/버튼류: a 또는 span
      genres = by('a, span', sec).map(text).filter(Boolean);
    }
    if (!genres.length) {
      // 헤더 보조 (페이지 상단 칩)
      genres = by('a[rel="genre"], span[rel="genre"]').map(text).filter(Boolean);
    }
    // 너무 긴 텍스트(문장) 제거 (장르 칩만 살리기)
    genres = genres.filter(g => g.length <= 20);
    res.genres = dedup(genres);
  })();

  // ---- 태그(테마) ----
  (() => {
    const sec = sectionByH2('작품 태그');
    let tags = [];
    if (sec) {
      tags = by('a span, a, span', sec).map(text).filter(Boolean);
    }
    tags = tags.map(t => t.replace(/^#/, '')).filter(t => t && t.toLowerCase() !== 'null');
    res.themes = dedup(tags);
  })();

  // ---- 줄거리 ----
  (() => {
    const sec = sectionByH2('줄거리');
    let syn = "";
    if (sec) {
      const body = sec.querySelector('p, .-mMZ9fV, .content, .body, .text, div');
      syn = body ? text(body) : text(sec);
    }
    if (/등록된 줄거리가 없습니다|줄거리 없음/.test(syn)) syn = "";
    // 상단에 짤막한 설명 박스가 있을 경우 보조
    if (!syn) {
      const headSyn = document.querySelector('.bnHDzeE, .summary, .lead, .desc');
      if (headSyn) {
        const s = text(headSyn);
        if (!/등록된 줄거리가 없습니다|줄거리 없음/.test(s)) syn = s;
      }
    }
    res.synopsis = syn;
  })();

  // ---- 캐릭터 및 성우 (최대 6명) ----
  (() => {
    const sec = sectionByH2('캐릭터 및 성우');
    let cast = [];
    if (sec) {
      const links = by('a[href*="/staff/"]', sec);
      for (const a of links) {
        // 카드 내부의 텍스트 라인을 전부 모으고, '주연/조연' 같은 역할 단어 제외 후 가장 긴 라인을 이름으로 추정
        const txt = text(a);
        let lines = txt.split(/\s*\n+\s*| {2,}/g).map(s=>s.trim()).filter(Boolean);
        lines = lines.filter(x => !/^(주연|조연)$/.test(x));
        lines.sort((x,y)=>y.length-x.length);
        if (lines[0]) cast.push(lines[0]);
      }
    }
    res.cast = dedup(cast).slice(0, 6);
  })();

  // ---- 작품 제작 (감독/애니메이션 제작/제작) ----
  (() => {
    const sec = sectionByH2('작품 제작');
    let director = '';
    const studios = [];
    const producers = [];
    if (sec) {
      // 카드 내 이름(.iO6bs1d 등) + 역할(작은 글씨)을 전부 텍스트로 묶어 역할 분류
      const cards = by('a, li, .card, ._1coMKET, ._2hRLd-G', sec);
      for (const el of cards) {
        const t = text(el);
        if (!t) continue;
        // 이름 후보: 줄 중 가장 긴 것, 역할 후보: 줄 중 '감독/애니메이션 제작/제작' 포함
        const lines = dedup(t.split(/\s*\n+\s*| {2,}/g).map(x=>x.trim()).filter(Boolean));
        const role = lines.find(x => /(감독|애니메이션\s*제작|제작)/.test(x)) || "";
        const name = lines.slice().sort((a,b)=>b.length-a.length)[0] || "";
        if (!name) continue;
        if (/감독/.test(role)) { if (!director) director = name; }
        else if (/애니메이션\s*제작/.test(role)) { studios.push(name); }
        else if (/제작/.test(role)) { producers.push(name); }
      }
    }
    res.director = director;
    res.studios = dedup(studios);
    res.producers = dedup(producers);
  })();

  return res;
}
"""


# -------------------- Playwright 네비게이션 --------------------
async def goto_resilient(page, url: str, nav_timeout_ms: int = 45000):
    """
    networkidle은 WebSocket/롱폴링 때문에 잘 안 끝난다.
    1) domcontentloaded → 2) load → 3) 기본 goto 순으로 시도.
    실패 시 None 리턴.
    """
    try:
        return await page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout_ms)
    except PlaywrightTimeoutError:
        pass
    try:
        return await page.goto(url, wait_until="load", timeout=nav_timeout_ms)
    except PlaywrightTimeoutError:
        pass
    try:
        return await page.goto(url, timeout=nav_timeout_ms)
    except PlaywrightTimeoutError:
        return None


# -------------------- 단건 스크랩 --------------------
async def scrape_one(
    url: str,
    nav_timeout_s: int,
    selector_timeout_s: int,
    debug: bool = False,
) -> Dict[str, str]:
    """
    단일 INFO 페이지에서 데이터 추출.
    제목이 비면 실패로 간주하여 빈 dict 반환 → [SKIP]
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent=UA,
            viewport={"width": 1360, "height": 2200},
            locale="ko-KR",
        )
        page = await ctx.new_page()

        url = ensure_info_url(url)
        resp = await goto_resilient(page, url, nav_timeout_ms=nav_timeout_s * 1000)
        if resp is None:
            await browser.close()
            return {}

        # 섹션 헤더가 보일 때까지 대기 + 지연 로드를 위한 스크롤
        try:
            await page.wait_for_selector("h2, h3", timeout=selector_timeout_s * 1000)
        except PlaywrightTimeoutError:
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(600)
            try:
                await page.wait_for_selector("h2, h3", timeout=6000)
            except PlaywrightTimeoutError:
                await browser.close()
                return {}

        data = await page.evaluate(JS_EXTRACT)

        if debug:
            Path("debug_info.html").write_text(await page.content(), encoding="utf-8")
            await page.screenshot(path="debug_info.png", full_page=True)

        await browser.close()

    title = (data.get("title") or "").strip()
    if not title:
        return {}

    # CSV 매핑
    row = {h: "" for h in CSV_HEADERS}
    row["제목"] = title
    row["분기"] = (data.get("season") or "").strip()
    row["장르"] = ", ".join(data.get("genres", []) or [])
    row["테마"] = ", ".join(data.get("themes", []) or [])
    syn = (data.get("synopsis") or "").strip()
    if any(ph in syn for ph in PLACEHOLDER_SYNOP):
        syn = ""
    row["공식 줄거리"] = syn

    cast = data.get("cast", []) or []
    for i in range(6):
        row[f"성우{i+1}"] = cast[i] if i < len(cast) else ""

    row["감독"] = (data.get("director") or "").strip()
    row["애니메이션 제작"] = ", ".join(data.get("studios", []) or [])
    row["제작"] = ", ".join(data.get("producers", []) or [])
    row["URL"] = url
    return row


# -------------------- 대상 열거 --------------------
def enumerate_targets(args) -> Iterable[Tuple[Optional[int], str]]:
    if args.url:
        yield None, ensure_info_url(args.url)
    if args.id is not None:
        yield args.id, build_info_url(args.id)
    if args.range:
        s, e = args.range
        for cid in range(s, e + 1):
            yield cid, build_info_url(cid)
    if args.ids_file:
        p = Path(args.ids_file)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.isdigit():
                    yield int(line), build_info_url(int(line))
                else:
                    yield None, ensure_info_url(line)


# -------------------- 실행 루프 --------------------
async def run(args):
    out_path = Path(args.out)
    wrote_header = out_path.exists() and out_path.stat().st_size > 0

    ok = 0
    skip = 0
    consec_err = 0

    with out_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if not wrote_header:
            w.writerow(CSV_HEADERS)

        for cid, url in enumerate_targets(args):
            time.sleep(max(0.0, args.delay))
            try:
                data = await scrape_one(
                    url=url,
                    nav_timeout_s=args.nav_timeout,
                    selector_timeout_s=args.selector_timeout,
                    debug=args.debug,
                )
            except Exception as e:
                # 예상치 못한 런타임 에러는 SKIP 처리
                print(f"[SKIP] {cid if cid is not None else url}  예외: {type(e).__name__}: {e}")
                data = {}

            tag = str(cid) if cid is not None else url
            if not data:
                print(f"[SKIP] {tag}  INFO 추출 실패")
                skip += 1
                consec_err += 1
                if args.stop_after_errors and consec_err >= args.stop_after_errors:
                    print(f"\n[중단] 연속 오류 {consec_err}회로 중지합니다.")
                    break
                continue

            data["컨텐츠ID"] = str(cid) if cid is not None else ""
            w.writerow([data.get(h, "") for h in CSV_HEADERS])

            print(f"[OK]   {cid if cid is not None else '-'}  {data['제목']}")
            ok += 1
            consec_err = 0

    print(f"\n[완료] OK={ok}  SKIP={skip}  → {out_path.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="AniLIFE INFO Scraper (Playwright, [OK]/[SKIP])")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--url", help="단일 작품 URL")
    grp.add_argument("--id", type=int, help="단일 컨텐츠 ID")
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), help="ID 범위 스캔")
    ap.add_argument("--ids-file", help="ID/URL 목록 파일(줄바꿈 구분)")

    ap.add_argument("-o", "--out", default="anilife_info.csv", help="CSV 출력 경로")
    ap.add_argument("--delay", type=float, default=0.5, help="요청 간 대기(초)")
    ap.add_argument("--nav-timeout", type=int, default=45, help="페이지 이동 타임아웃(초)")
    ap.add_argument("--selector-timeout", type=int, default=12, help="요소 대기 타임아웃(초)")
    ap.add_argument("--stop-after-errors", type=int, default=0, help="연속 오류 N회 시 중단(0=비활성)")
    ap.add_argument("--debug", action="store_true", help="debug_info.html/png 저장")

    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
