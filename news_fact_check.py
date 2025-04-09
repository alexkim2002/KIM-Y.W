#########################
# 1) Import & Settings
#########################

import os
import re
import requests
import openai
import streamlit as st
from dotenv import load_dotenv # .env 파일 로드
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from datetime import datetime
from urllib.parse import urlparse, quote_plus
from bs4 import BeautifulSoup
from newspaper import Article

# .env 파일 로드
load_dotenv()

# (옵션) googlesearch 활용 시
# !pip install googlesearch-python
try:
    from googlesearch import search
except ImportError:
    search = None  # 설치 안 된 경우 대처용

#########################
# 2) Global Config
#########################

# OpenAI API 키 설정 (환경 변수에서 가져오기)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요.")
    st.stop()

# Google Gemini API 키 설정
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY", "f438158f8d45f67aede74b4a639e23a8365974ce")
genai.configure(api_key=GEMINI_API_KEY)

# 도메인별 신뢰도 점수 (예시)
DOMAIN_TRUST_SCORES = {
    "yonhapnews.co.kr": 95,
    "donga.com": 85,
    "chosun.com": 80,
    "seoul.co.kr": 90,
    "kmib.co.kr": 85,
    "news.sbs.co.kr": 75,
    "news.kbs.co.kr": 80,
    "mbc.co.kr": 80,
    "jtbc.joins.com": 75,
    "news.mt.co.kr": 70,
    "news1.kr": 70,
    "newsis.com": 70,
    "newstapa.org": 85,
    "factcheck.kr": 90
}

#########################
# 3) Utility Functions
#########################

def fetch_article(url: str) -> dict:
    """
    주어진 기사 URL로부터 본문, 제목, 발행일, 도메인을 추출하여 반환.
    newspaper3k를 사용하여 더 정확한 본문 추출.
    """
    try:
        article = Article(url, language='ko')
        article.download()
        article.parse()
        
        # 발행일이 없는 경우 BeautifulSoup으로 시도
        if not article.publish_date:
            resp = requests.get(url, timeout=10)
            html = resp.text
            date_match = re.search(r'(20\d{2}[-./]\d{1,2}[-./]\d{1,2})', html)
            if date_match:
                try:
                    article.publish_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
                except ValueError:
                    pass

        return {
            "title": article.title or "",
            "text": article.text or "",
            "date": article.publish_date,
            "domain": urlparse(url).netloc
        }
    except Exception as e:
        st.error(f"기사를 가져오는 중 오류 발생: {str(e)}")
        return {
            "title": "",
            "text": "",
            "date": None,
            "domain": urlparse(url).netloc
        }

def summarize_article(article_text: str, max_paragraphs: int = 3) -> str:
    """
    기사 본문을 GPT-4 기반 모델로 요약하여 반환.
    max_paragraphs로 문단 수를 제한.
    """
    if not article_text:
        return "요약할 텍스트가 없습니다."
    
    prompt = (
        f"다음 기사를 {max_paragraphs}문단 이내로 한국어 요약해줘:\n\n{article_text[:3500]}"
        "\n\n(반드시 요약은 지정된 문단 수 이내로만 작성해주세요.)"
    )
    
    try:
        # 최신 OpenAI API 형식으로 업데이트
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        st.error(f"요약 중 오류 발생: {str(e)}")
        return "요약 중 오류가 발생했습니다."

def score_domain_trust(domain: str) -> int:
    """
    기사 출처(도메인)에 대한 사전 정의 신뢰도 점수를 반환.
    알려지지 않은 도메인은 50점으로 처리.
    """
    base_domain = domain.lower()
    return DOMAIN_TRUST_SCORES.get(base_domain, 50)

def score_date_recency(publish_date) -> int:
    """
    기사 발행일을 기준으로 최신 기사일수록 높은 점수를, 
    발행일 정보가 없으면 0점을 부여.
    """
    if not publish_date:
        return 0
    
    # datetime.datetime 타입을 datetime.date 타입으로 변환
    today = datetime.now().date()
    if isinstance(publish_date, datetime):
        publish_date = publish_date.date()
    
    days_diff = (today - publish_date).days
    if days_diff < 0:
        # 미래 날짜면 0점
        return 0
    elif days_diff <= 1:
        return 100  # 하루 이내
    elif days_diff <= 7:
        return 90   # 1주일 이내
    elif days_diff <= 30:
        return 70   # 1달 이내
    elif days_diff <= 365:
        return 50   # 1년 이내
    else:
        return 30   # 1년 초과

def extract_claims(text: str, max_claims: int = 5):
    """
    본문에서 인용부호(“...”) 안에 있는 문장을 최대 max_claims개까지 추출.
    """
    claims = re.findall(r'“([^"]+)”', text)
    claims = [c.strip() for c in claims if len(c.strip()) > 0]
    if len(claims) > max_claims:
        claims = claims[:max_claims]
    return claims

def search_web(query: str, num_results: int = 5):
    """
    Google 검색을 통해 query와 관련된 URL을 최대 num_results개 반환.
    (googlesearch 설치 필요)
    """
    if search is None:
        return []
    results = []
    try:
        for url in search(query, lang='ko', num=num_results, stop=num_results):
            results.append(url)
    except Exception:
        pass
    return results

def fetch_snippet(url: str) -> str:
    """
    주어진 URL을 requests로 가져와 title과 meta description(또는 p 태그 일부)을 연결해 반환.
    """
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.title.string.strip() if soup.title else ""
        meta_desc = soup.find('meta', attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            snippet = meta_desc["content"]
        else:
            # 첫 번째 p 태그 일부
            p = soup.find('p')
            snippet = p.get_text(" ", strip=True)[:150] if p else ""
        return f"{title} - {snippet}"
    except:
        return ""

# Google Fact Check API 관련 코드!!!
# 기존 get_factcheck_results와 verify_claim_with_factcheck_api 함수 제거
# def get_factcheck_results(claim: str) -> list:
# def verify_claim_with_factcheck_api(claim: str) -> dict:

def verify_claim_with_gemini(claim: str, evidence_list: list) -> dict:
    """
    Google Gemini API를 사용하여 주장에 대한 팩트체크를 수행합니다.
    """
    try:
        # 증거 텍스트 준비
        evidence_text = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(evidence_list)])
        
        # 프롬프트 작성
        prompt = f"""당신은 검증된 정보를 바탕으로 팩트체크를 수행하는 전문가입니다.
다음 주장을 검증하고 사실 여부를 판단해주세요.

주장: "{claim}"

아래는 이 주장과 관련된 검색 결과입니다:
{evidence_text}

위 정보를 바탕으로, 다음 형식으로 주장의 사실 여부를 평가해주세요:
1. 판정 - "사실", "거짓", "부분 사실", "불확실" 중 하나를 선택
2. 이유 - 판정 이유를 간략히 설명 (2-3 문장)

답변 형식: "판정: 판정 이유"
예: "사실: 다수의 신뢰할 수 있는 출처에서 확인된 정보입니다."
"""
        
        # Gemini 모델 설정
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 300,
        }

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Gemini API 호출
        model = genai.GenerativeModel(
            model_name="gemini-1.0-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        response = model.generate_content(prompt)
        verdict = response.text.strip()
        
        # 판정 결과 반환
        return {
            "verdict": verdict,
            "is_from_gemini": True
        }
    except Exception as e:
        st.error(f"Gemini API 호출 중 오류 발생: {str(e)}")
        return {
            "verdict": f"불확실: Gemini API 오류 - {str(e)}",
            "is_from_gemini": False
        }

# 기존 get_factcheck_results 함수를 대체하는 함수
def get_factcheck_with_gemini(claim: str) -> dict:
    """
    주장에 대해 웹 검색 후 Gemini API로 팩트체크를 수행합니다.
    """
    try:
        # 웹 검색으로 관련 정보 수집
        query = f"{claim} 사실 여부"
        result_urls = search_web(query, num_results=5)
        snippets = [fetch_snippet(u) for u in result_urls if u]
        
        if not snippets:
            return {
                "verdict": "불확실: 관련 정보를 충분히 찾지 못했습니다.",
                "is_from_gemini": False
            }
        
        # Gemini로 팩트체크 수행
        result = verify_claim_with_gemini(claim, snippets)
        return result
        
    except Exception as e:
        st.error(f"팩트체크 중 오류 발생: {str(e)}")
        return {
            "verdict": "불확실 (처리 오류)",
            "is_from_gemini": False
        }

# Gemini 결과를 바탕으로 신뢰도 점수 계산
def score_content_trust(claim_verifications: dict) -> int:
    """
    Gemini가 판정한 '사실/거짓/불확실/부분 사실' 결과를 종합해 기사 내용 신뢰도(0~100) 산정.
    - 사실: +1
    - 거짓: -0.2 총점에서 20점 감산
    - 불확실: 0.5로 처리 
    - 부분 사실: 0.7로 처리
    """
    if not claim_verifications:
        return 50  # 검증할 주장 없음 → 중립 점수

    results = list(claim_verifications.values())
    # 사실 판정
    fact_count = sum("사실:" in r or "사실 " in r or r.startswith("사실") for r in results)
    # 거짓 판정
    false_count = sum("거짓:" in r or "거짓 " in r or r.startswith("거짓") for r in results)
    # 불확실 판정
    uncertain_count = sum("불확실:" in r or "불확실 " in r or r.startswith("불확실") for r in results)
    # 부분 사실 판정
    partial_count = sum("부분 사실:" in r or "부분 사실 " in r or "부분사실" in r or r.startswith("부분 사실") for r in results)

    total_claims = len(results)

    # 사실 비율
    fact_ratio = fact_count / total_claims
    # 부분 사실 점수 (70%)
    partial_ratio = 0.7 * partial_count / total_claims
    # 기본 점수 계산
    base_score = (fact_ratio + partial_ratio) * 100.0
    # 페널티 계산
    false_penalty = false_count * 20
    uncertain_penalty = uncertain_count * 10

    final_score = base_score - false_penalty - uncertain_penalty
    if final_score < 0:
        final_score = 0
    if final_score > 100:
        final_score = 100
    return int(final_score)

#########################
# 4) Streamlit App
#########################

def main():
    st.set_page_config(
        page_title="뉴스 기사 신뢰도 평가기",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 뉴스 기사 신뢰도 평가기")
    st.markdown("""
    이 앱은 뉴스 기사의 신뢰도를 다음과 같은 기준으로 평가합니다:
    - 출처 신뢰도 (30%): 뉴스 매체의 신뢰도
    - 날짜 신뢰도 (20%): 기사의 최신성
    - 내용 신뢰도 (50%): 기사 내용의 사실 여부
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url_input = st.text_input("기사 URL 입력:", placeholder="https://example.com/news/article")
        
        if st.button("평가하기", type="primary"):
            if not url_input.strip():
                st.error("유효한 URL을 입력해주세요.")
                return
            
            with st.spinner("기사를 가져오는 중..."):
                article_data = fetch_article(url_input)
            
            if not article_data["text"]:
                st.error("기사를 불러오지 못했습니다. URL이 유효한지 확인해주세요.")
                return
            
            # 1) 요약
            with st.spinner("기사를 요약하는 중..."):
                summary = summarize_article(article_data["text"])

            # 2) 출처 신뢰도
            domain_score = score_domain_trust(article_data["domain"])
            # 3) 날짜 신뢰도
            date_score = score_date_recency(article_data["date"])
            # 4) 주요 주장 추출
            claims = extract_claims(article_data["text"], max_claims=5)

            # 5) 웹 검색 & 검증
            claim_verifications = {}
            if claims:
                with st.spinner("주요 주장들을 검증하는 중..."):
                    for c in claims:
                        # Google Gemini를 활용한 팩트체크
                        result = get_factcheck_with_gemini(c)
                        claim_verifications[c] = result["verdict"]
                        
                        # Gemini에서 검증된 결과인 경우 표시
                        if result["is_from_gemini"]:
                            st.info(f"주장 \"{c}\"은(는) Google Gemini AI로 검증되었습니다.")

            # 6) 내용 신뢰도 점수
            content_score = score_content_trust(claim_verifications)

            # 7) 종합 점수
            overall_score = int(domain_score*0.3 + date_score*0.2 + content_score*0.5)

            # 결과 출력
            st.subheader("📝 기사 요약")
            st.write(summary)

            # 신뢰도 점수 시각화
            st.subheader("📊 신뢰도 평가")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("출처 신뢰도", f"{domain_score}/100", f"{domain_score-50:+d}")
            with col2:
                st.metric("날짜 신뢰도", f"{date_score}/100", f"{date_score-50:+d}")
            with col3:
                st.metric("내용 신뢰도", f"{content_score}/100", f"{content_score-50:+d}")
            with col4:
                st.metric("종합 신뢰도", f"{overall_score}/100", f"{overall_score-50:+d}")
            
            # 주장별 검증 결과
            if claim_verifications:
                st.subheader("🔍 검증된 주요 주장들")
                for claim, result in claim_verifications.items():
                    st.markdown(f"- **\"{claim}\"** → {result}")

            st.success("평가가 완료되었습니다.")
    
    with col2:
        st.subheader("ℹ️ 사용 안내")
        st.markdown("""
        1. 뉴스 기사의 URL을 입력창에 붙여넣기 합니다.
        2. '평가하기' 버튼을 클릭합니다.
        3. 잠시 후 기사의 요약과 신뢰도 평가 결과가 표시됩니다.
        
        **참고**: 
        - 평가에는 몇 분 정도 소요될 수 있습니다.
        - 인터넷 연결이 필요합니다.
        - OpenAI API 키가 필요합니다.
        """)

# Streamlit에서 실행
if __name__ == "__main__":
    main()
