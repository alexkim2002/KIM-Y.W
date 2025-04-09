#########################
# 1) Import & Settings
#########################

import os
import re
import requests
import openai
import streamlit as st
from dotenv import load_dotenv # .env 파일 로드

from datetime import datetime
from urllib.parse import urlparse
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

def verify_claim_with_llm(claim: str, evidence_list: list) -> str:
    """
    claim(주장)과 evidence_list(검색으로 모은 각 페이지의 간단 요약)을 바탕으로,
    GPT-4에 사실 여부 판정을 요청. 
    결과 문자열 예: "사실: ...", "거짓: ...", "불확실: ..."
    """
    if not evidence_list:
        # 근거가 전혀 없으면 "불확실"로 처리
        return "불확실 (검색 결과 없음)"
    
    evidence_text = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(evidence_list)])
    prompt = (
        "당신은 사실 관계를 평가해주는 전문가입니다. "
        "사용자가 제시한 주장과 검색 결과를 보고 사실 여부를 판단해주세요.\n\n"
        f"주장: \"{claim}\"\n\n"
        "[검색 결과]\n"
        f"{evidence_text}\n\n"
        "위 정보를 참고하여 아래 형식으로 답변:\n"
        "1) \"사실\", \"거짓\", \"불확실\" 중 하나.\n"
        "2) 짧은 사유.\n"
        "(예: \"사실: 여러 기사가 일치한다고 보도하고 있습니다.\")"
    )
    try:
        # 최신 OpenAI API 형식으로 업데이트
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        st.error(f"판정 중 오류 발생: {str(e)}")
        return "불확실 (LLM 오류)"

def score_content_trust(claim_verifications: dict) -> int:
    """
    LLM이 판정한 '사실/거짓/불확실' 결과를 종합해 기사 내용 신뢰도(0~100) 산정.
    - 사실: +1
    - 거짓: -0.2 총점에서 20점 감산 (단순 예시)
    - 불확실: 0.5로 처리 
    """
    if not claim_verifications:
        return 50  # 검증할 주장 없음 → 중립 점수

    results = list(claim_verifications.values())
    # 사실 판정
    fact_count = sum("사실" in r for r in results)
    # 거짓 판정
    false_count = sum("거짓" in r for r in results)
    # 불확실 판정
    uncertain_count = sum("불확실" in r for r in results)

    total_claims = len(results)

    # 사실 비율
    fact_ratio = fact_count / total_claims  
    # 단순하게 fact_ratio * 100 (기본점수)에서 거짓 당 20점씩 빼기
    base_score = fact_ratio * 100.0
    penalty = false_count * 20

    # 불확실한 주장들은 사실/거짓 중간 정도라 가정해 보정 가능(옵션)
    # 예를 들어 불확실 1개당 -10점 하는 등...
    uncertain_penalty = uncertain_count * 10

    final_score = base_score - penalty - uncertain_penalty
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
            if claims and search:
                with st.spinner("주요 주장들을 검증하는 중..."):
                    for c in claims:
                        # 검색
                        query = f"{c} 사실 여부"
                        result_urls = search_web(query, num_results=3)
                        # 각 URL에 대해 snippet 추출
                        snippets = [fetch_snippet(u) for u in result_urls if u]
                        # LLM 검증
                        verdict = verify_claim_with_llm(c, snippets)
                        claim_verifications[c] = verdict

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
