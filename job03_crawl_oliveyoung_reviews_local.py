"""
[Job 03] 올리브영 제품 리뷰 크롤링 (무제한 수집 모드)
========================================================
실행 순서: 3번 (job02 이후)
입력: datasets/oliveyoung_product_list.csv  (job02 결과)
출력: datasets/oliveyoung_reviews.csv
      컬럼: product_name, star, skin_type, review

job02에서 수집한 제품 목록을 기반으로, 각 제품 상세 페이지의
리뷰 탭을 열어 모든 페이지의 리뷰를 수집한다.

특징:
- Shadow DOM(oy-review-*) 기반 올리브영 리뷰 구조 지원
- 20개 제품마다 브라우저를 재시작해 메모리 과부하 방지
- 이미 수집된 제품은 건너뛰는 이어받기 기능
- navigator.webdriver 속성 제거로 봇 탐지 우회
"""

from playwright.sync_api import sync_playwright
import csv
import time
import os
import random
import re


def crawl_reviews_stealth():
    input_file  = "./datasets/oliveyoung_product_list.csv"
    output_file = "./datasets/oliveyoung_reviews.csv"

    # ── 입력 파일 확인 ──────────────────────────────────────
    if not os.path.exists(input_file):
        print("❌ 제품 목록 파일이 없습니다. job02를 먼저 실행하세요.")
        return

    # 제품 목록 로드
    products = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader   = csv.DictReader(f)
        products = list(reader)

    # ── 이어받기: 이미 완료된 제품 건너뜀 ──────────────────
    # 기존 output_file에서 수집된 product_name 목록을 읽어 중복 방지
    processed_products = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_products.add(row['product_name'])

    # 아직 처리되지 않은 제품만 추출
    target_products = [p for p in products if p['product_name'] not in processed_products]

    if not target_products:
        print("✅ 모든 제품 수집이 이미 완료되었습니다.")
        return

    print(f"📦 총 {len(target_products)}개의 제품 수집을 시작합니다.")

    # ── 20개 단위로 브라우저 재시작 ────────────────────────
    # Playwright 브라우저는 장시간 실행 시 메모리 누수가 발생하므로
    # 20개 제품마다 브라우저를 새로 시작한다.
    batch_size = 20
    for i in range(0, len(target_products), batch_size):
        batch = target_products[i:i + batch_size]
        print(f"\n🔄 브라우저 세션 시작 "
              f"({i + 1} ~ {min(i + batch_size, len(target_products))} 번째 제품)")

        with sync_playwright() as p:
            # headless=False: 로컬 실행 시 브라우저 창으로 진행 상황 확인 가능
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                viewport={'width': 1280, 'height': 1024},
            )
            page = context.new_page()

            # navigator.webdriver = undefined 로 설정 — Selenium/Playwright 탐지 우회
            page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            # ── 파일을 append 모드로 열어 이어서 저장 ──────────
            with open(output_file, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["product_name", "star", "skin_type", "review"]
                )
                # 파일이 새로 생성된 경우에만 헤더 작성
                if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                    writer.writeheader()

                for prod in batch:
                    print(f"\n👉 {prod['product_name']} 분석 중...")
                    try:
                        # 제품 상세 페이지의 리뷰 탭으로 직접 이동
                        review_url = f"{prod['product_link']}&tab=review"
                        page.goto(review_url, wait_until="load", timeout=90000)
                        # 동적 렌더링 대기 (5~8초 랜덤 — 봇 탐지 방지)
                        time.sleep(random.uniform(5, 8))

                        # ── 리뷰 유무 확인 ──────────────────────────
                        total_count_text = "0"
                        try:
                            # Shadow DOM 내부의 총 리뷰 건수 확인
                            count_el = page.locator(".total-count").first
                            if count_el.is_visible():
                                total_count_text = count_el.inner_text()
                        except:
                            pass

                        if "0건" in total_count_text:
                            print("  ℹ️ 리뷰가 없는 상품입니다. 스킵.")
                            continue

                        # ── 페이지네이션 루프: 마지막 페이지까지 수집 ──
                        p_idx = 1
                        while True:
                            # 스크롤 다운 — Shadow DOM 리뷰 아이템 렌더링 유도
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            time.sleep(2)

                            # oy-review-review-item: 올리브영 Shadow DOM 리뷰 컴포넌트
                            items = page.locator("oy-review-review-item").all()
                            if not items:
                                print(f"    ⚠️ Page {p_idx}: 리뷰 아이템 로드 실패. (재시도 중...)")
                                time.sleep(3)
                                items = page.locator("oy-review-review-item").all()

                            if not items:
                                break  # 재시도 후에도 없으면 수집 종료

                            count = 0
                            for item in items:
                                try:
                                    # 리뷰 본문 추출
                                    content = item.locator(
                                        "oy-review-review-content p"
                                    ).inner_text(timeout=5000)

                                    # 작성자 피부 정보 추출 (예: "지성 / 밝은 피부")
                                    skin_info_elements = item.locator(
                                        "oy-review-review-user .skin-type"
                                    ).all()
                                    skin_info = " / ".join(
                                        [el.inner_text() for el in skin_info_elements]
                                    )

                                    # 10자 이상인 리뷰만 저장 (광고성 단문 필터링)
                                    if len(content.strip()) > 10:
                                        writer.writerow({
                                            "product_name": prod['product_name'],
                                            "star":         "5",   # 별점은 기본값 5 사용
                                            "skin_type":    skin_info,
                                            "review":       content.strip(),
                                        })
                                        count += 1
                                except:
                                    continue

                            print(f"    Page {p_idx}: {count}개 리뷰 저장")

                            # ── 다음 페이지 이동 ────────────────────
                            try:
                                next_p   = p_idx + 1
                                next_btn = page.locator(f"a[data-page-no='{next_p}']")

                                # 10페이지 단위 넘김 버튼 대응
                                if not next_btn.is_visible():
                                    next_arrow = page.locator(
                                        "button[class*='next'], .pagination-next"
                                    ).first
                                    if next_arrow.is_visible():
                                        next_arrow.click()
                                        time.sleep(3)
                                        next_btn = page.locator(f"a[data-page-no='{next_p}']")

                                if next_btn.is_visible():
                                    next_btn.click()
                                    p_idx += 1
                                    time.sleep(random.uniform(2, 4))
                                else:
                                    print("    🏁 마지막 페이지입니다.")
                                    break
                            except:
                                break

                        f.flush()  # 제품 단위로 즉시 파일에 기록 (중단 대비)

                    except Exception as e:
                        print(f"  ❌ 상품 처리 중 오류 (스킵): {e}")
                        continue

            browser.close()
            # 배치 완료 후 10초 대기 — IP 차단 방지
            print(f"☕ 한 세트 완료. 차단 방지를 위해 10초 휴식...")
            time.sleep(10)

    print("\n✨ [임무 완료] 모든 카테고리 제품의 모든 리뷰를 수집했습니다!")


if __name__ == "__main__":
    crawl_reviews_stealth()
