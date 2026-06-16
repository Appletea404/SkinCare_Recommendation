"""
[Job 02] 올리브영 제품 목록 크롤링
=====================================
실행 순서: 2번 (job01 이후)
입력: 없음 (올리브영 웹사이트에서 직접 수집)
출력: datasets/oliveyoung_product_list.csv
      컬럼: category, product_brand, product_name, product_link

Playwright로 올리브영 카테고리 페이지를 열어 제품명·브랜드·링크를 수집한다.
수집된 목록은 web_app.py의 '분석 키워드 연관 제품' 추천에 사용된다.
"""

from playwright.sync_api import sync_playwright
import csv
import time
from bs4 import BeautifulSoup
import os

# ─────────────────────────────────────────────────────────
# 올리브영 카테고리 번호 매핑
# dispCatNo 파라미터 값은 올리브영 카테고리 URL에서 확인
# ─────────────────────────────────────────────────────────
CATEGORY_MAP = {
    # ── 스킨케어 ──
    "스킨/토너":        "100000100010013",
    "에센스/세럼/앰플": "100000100010014",
    "크림":             "100000100010015",
    "로션":             "100000100010016",
    "미스트/오일":      "100000100010011",
    "스킨케어 세트":    "100000100010004",
    "스킨케어 디바이스":"100000100010010",

    # ── 메이크업 ──
    "립메이크업":   "100000100020006",
    "베이스메이크업":"100000100020001",
    "아이메이크업": "100000100020007",
}


def get_product_list(playwright_page, cat_no, page):
    """
    올리브영 카테고리 목록 페이지 HTML을 반환한다.
    - cat_no: 카테고리 번호
    - page: 페이지 번호 (1부터 시작)
    - 로딩 완료 후 3.5초 대기 — JS 렌더링 시간 확보
    """
    url = (
        f"https://www.oliveyoung.co.kr/store/display/getMCategoryList.do"
        f"?dispCatNo={cat_no}&pageIdx={page}&rowsPerPage=24"
    )
    print(f"Fetching: Category {cat_no}, Page {page}")
    try:
        playwright_page.goto(url, wait_until="load", timeout=60000)
        time.sleep(3.5)
        return playwright_page.content()
    except Exception as e:
        print(f"Error: {e}")
        return None


def parse_product_list(html):
    """
    HTML에서 제품 정보를 파싱해 리스트로 반환한다.
    - .prd_info 셀렉터로 각 제품 카드 선택
    - 브랜드(.tx_brand), 제품명(.tx_name), 링크(a[href]) 추출
    - 상대 경로 링크는 절대 경로로 변환
    """
    if not html:
        return []
    soup = BeautifulSoup(html, 'lxml')
    items = soup.select(".prd_info")
    data = []
    for item in items:
        try:
            brand    = item.select_one(".tx_brand").get_text(strip=True)
            name     = item.select_one(".tx_name").get_text(strip=True)
            raw_link = item.select_one("a")["href"]
            link = raw_link if raw_link.startswith("http") \
                else f"https://www.oliveyoung.co.kr{raw_link}"
            data.append({
                "product_brand": brand,
                "product_name":  name,
                "product_link":  link,
            })
        except:
            continue
    return data


def crawl_products():
    """
    CATEGORY_MAP의 모든 카테고리를 순회하며 제품 목록을 수집한다.
    PAGES_PER_CATEGORY: 카테고리당 수집할 페이지 수 (1페이지 = 24개)
    결과를 oliveyoung_product_list.csv로 저장한다.
    """
    PAGES_PER_CATEGORY = 2  # 카테고리당 2페이지(최대 48개) 수집
    total_data = []

    with sync_playwright() as p:
        # headless=True: 브라우저 창을 띄우지 않고 백그라운드 실행
        browser = p.chromium.launch(headless=True)
        # User-Agent를 일반 브라우저처럼 설정해 차단 방지
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for cat_name, cat_no in CATEGORY_MAP.items():
            print(f"\nProcessing Category: {cat_name}")
            for p_idx in range(1, PAGES_PER_CATEGORY + 1):
                html  = get_product_list(page, cat_no, p_idx)
                items = parse_product_list(html)
                if not items:  # 더 이상 제품이 없으면 다음 카테고리로
                    break
                for item in items:
                    item['category'] = cat_name  # 카테고리명 추가
                total_data.extend(items)

        browser.close()

    # CSV 저장 (utf-8-sig: 엑셀에서 한글 깨짐 방지)
    os.makedirs("./datasets", exist_ok=True)
    with open("./datasets/oliveyoung_product_list.csv", "w",
              newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "product_brand", "product_name", "product_link"]
        )
        writer.writeheader()
        writer.writerows(total_data)

    print(f"\nFinished! Total products collected: {len(total_data)}")


if __name__ == "__main__":
    crawl_products()
