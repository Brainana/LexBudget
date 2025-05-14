from playwright.sync_api import sync_playwright
import argparse  # Used to parse arguments
import logging  # Used to log errors

# usage:
# python web-2-pdf.py --url https://www.lhsproject.lexingtonma.org/projectfaqs  --pdfFileName lhs_project_faqs.pdf
# python web-2-pdf.py --url https://www.lhsproject.lexingtonma.org/community-submissions  --pdfFileName lhs_project_community_submissions.pdf
# python web-2-pdf.py --url https://www.yes4lex.org/cec-report  --pdfFileName lhs_project_cec_report.pdf
# python web-2-pdf.py --url https://lexobserver.org/2024/10/17/inside-the-the-hot-stuffy-overcrowded-lexington-high-school/ --expandingButtons false --pdfFileName lhs_project_inside_hot_stuffy_overcrowded_lhs.pdf
# python web-2-pdf.py --url https://lexobserver.org/2024/10/16/lexington-high-school-is-old-and-run-down-here-are-some-photos-that-show-it/ --expandingButtons false --pdfFileName lhs_project_old_rundown.pdf
# python web-2-pdf.py --url https://lexobserver.org/2024/10/17/how-years-of-construction-at-lhs-will-affect-its-students/  --pdfFileName lhs_project_construction_years.pdf
# python web-2-pdf.py --url https://lexobserver.org/2024/10/18/what-sustainability-features-are-in-the-works-for-the-new-high-school-building-and-are-they-worth-the-cost/  --pdfFileName lhs_project_sustain_features.pdf
# python web-2-pdf.py --url https://lexobserver.org/2024/10/09/letter-to-the-editor-a-new-high-school-to-meet-the-educational-needs-of-future-generations/ --pdfFileName lhs_project_future_gen.pdf
# python web-2-pdf.py --url https://www.yes4lex.org/march-2025-newsletter --pdfFileName lhs_project_2025_03_news_letter.pdf
# python web-2-pdf.py --url https://www.yes4lex.org/2025-02-26-newsletter  --pdfFileName lhs_project_2025_02_news_letter.pdf
# python web-2-pdf.py --url https://lexobserver.org/2024/10/17/explaining-the-six-concepts-for-the-new-lhs-and-which-are-front-runners/  --pdfFileName lhs_project_6_concepts.pdf


# Configure the logging
logging.basicConfig(
    filename="record_docs_scraper_errors.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)

# Get all the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--url", type=str, required=True)
parser.add_argument("--expandingButtons", type=bool, required=True)
parser.add_argument("--pdfFileName", type=str, required=True)

args = parser.parse_args()

def convert_website_2_pdf( website_url, expanding_buttons,pdf_file_name):
    with sync_playwright() as p:
        print("loading page ...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(website_url)

        # Click all buttons
        if expanding_buttons :
            buttons = page.query_selector_all("button")
            for btn in buttons:
                try:
                    btn.click()
                    print("expanding button ...")
                except:
                    pass

        # Save as PDF
        page.pdf(path=pdf_file_name)
        print("saving to pdf  ...")
        browser.close()

convert_website_2_pdf (args.url, args.expandingButtons, args.pdfFileName)