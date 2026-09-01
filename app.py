
import streamlit as st
import pandas as pd
import numpy as np
import re
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF reader
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening & Job Matching",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #bbbbbb;
}

.score-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    background: #151922;
}

.strong {
    color: #2ecc71;
    font-weight: bold;
}

.moderate {
    color: #f1c40f;
    font-weight: bold;
}

.low {
    color: #e74c3c;
    font-weight: bold;
}

.skill-match {
    background-color: #123d2b;
    padding: 14px;
    border-radius: 10px;
}

.skill-missing {
    background-color: #4a4012;
    padding: 14px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# JOB ROLES
# ============================================================

JOB_ROLES = {

    "Python Developer": """
    We are looking for a Python Developer with strong knowledge of Python,
    SQL, Pandas, NumPy, Git, GitHub and software development.
    Experience with Flask or Django is preferred.
    Good problem solving, communication and teamwork skills are required.
    Bachelor's degree in Computer Science, Information Technology or related field.
    """,

    "Data Scientist": """
    We are looking for a Data Scientist with Python, Machine Learning,
    Pandas, NumPy, Scikit-learn, SQL, Data Analysis and Statistics skills.
    Knowledge of Data Visualization, TensorFlow, NLP and Git is preferred.
    Bachelor's or Master's degree in Computer Science, Data Science,
    Artificial Intelligence or related field.
    """,

    "Machine Learning Engineer": """
    We are looking for a Machine Learning Engineer with Python,
    Machine Learning, Scikit-learn, TensorFlow, PyTorch, Pandas,
    NumPy, SQL and Git skills.
    Knowledge of Deep Learning, NLP and model deployment is preferred.
    Bachelor's degree in Computer Science, Artificial Intelligence,
    Data Science or related field.
    """,

    "Data Analyst": """
    We are looking for a Data Analyst with Python, SQL, Pandas, NumPy,
    Excel, Power BI, Data Analysis and Data Visualization skills.
    Good analytical and communication skills are required.
    Bachelor's degree in Computer Science, Information Technology,
    Statistics, Mathematics, Data Science or related field.
    """,

    "AI Engineer": """
    We are looking for an AI Engineer with Python, Artificial Intelligence,
    Machine Learning, Deep Learning, TensorFlow, PyTorch, NLP,
    Computer Vision, Pandas, NumPy and Git skills.
    Bachelor's or Master's degree in Artificial Intelligence,
    Computer Science, Data Science or related field.
    """
}


# ============================================================
# SKILL DATABASE
# ============================================================

SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "mysql",
    "mongodb",
    "pandas",
    "numpy",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "data visualization",
    "tensorflow",
    "pytorch",
    "nlp",
    "natural language processing",
    "computer vision",
    "opencv",
    "power bi",
    "excel",
    "git",
    "github",
    "flask",
    "django",
    "react",
    "javascript",
    "html",
    "css",
    "statistics",
    "problem solving",
    "communication",
    "teamwork"
]


# ============================================================
# EDUCATION KEYWORDS
# ============================================================

EDUCATION_KEYWORDS = [
    "b.tech",
    "btech",
    "b.e",
    "be ",
    "b.sc",
    "bsc",
    "bca",
    "mca",
    "m.tech",
    "mtech",
    "m.sc",
    "msc",
    "computer science",
    "information technology",
    "data science",
    "artificial intelligence",
    "electronics",
    "engineering"
]


# ============================================================
# FUNCTIONS
# ============================================================

def extract_pdf_text(uploaded_file):

    if PdfReader is None:
        st.error(
            "PDF library is missing. Install pypdf using: "
            "pip install pypdf"
        )
        return ""

    try:
        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    except Exception as e:
        st.error(f"Could not read {uploaded_file.name}: {e}")
        return ""


def normalize(text):

    text = text.lower()

    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_skills(text):

    normalized = normalize(text)

    found = []

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        if re.search(pattern, normalized):

            found.append(skill)

    return sorted(set(found))


def extract_email(text):

    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    if match:
        return match.group(0)

    return "Not detected"


def extract_phone(text):

    match = re.search(
        r"(?:\+91[\s-]?)?[6-9]\d{9}",
        text
    )

    if match:
        return match.group(0)

    return "Not detected"


def extract_name(text, filename):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:

        first_line = lines[0]

        if (
            len(first_line) < 60
            and "@" not in first_line
            and not re.search(r"\d{5,}", first_line)
        ):
            return first_line

    return Path(filename).stem.replace("_", " ").title()


def extract_education(text):

    normalized = normalize(text)

    found = []

    education_patterns = [
        "b.tech",
        "btech",
        "b.e",
        "b.sc",
        "bsc",
        "bca",
        "mca",
        "m.tech",
        "mtech",
        "m.sc",
        "msc",
        "computer science",
        "information technology",
        "data science",
        "artificial intelligence",
        "electronics and communication"
    ]

    for item in education_patterns:

        if item in normalized:

            found.append(item.upper())

    return sorted(set(found))


def extract_experience(text):

    normalized = normalize(text)

    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:year|years|yr|yrs)",
        normalized
    )

    if matches:

        values = []

        for value in matches:

            try:
                values.append(float(value))
            except:
                pass

        if values:
            return max(values)

    if "internship" in normalized or "intern" in normalized:
        return 0.5

    return 0.0


def calculate_nlp_score(resume_text, job_text):

    resume_text = normalize(resume_text)
    job_text = normalize(job_text)

    if not resume_text or not job_text:
        return 0

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            [resume_text, job_text]
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0] * 100

        return round(float(score), 2)

    except:
        return 0


def calculate_skill_score(resume_skills, required_skills):

    if not required_skills:
        return 0

    matched = set(resume_skills).intersection(
        set(required_skills)
    )

    return round(
        len(matched) / len(required_skills) * 100,
        2
    )


def calculate_education_score(resume_text, job_text):

    resume = normalize(resume_text)
    job = normalize(job_text)

    resume_education = set()
    job_education = set()

    for item in EDUCATION_KEYWORDS:

        if item in resume:
            resume_education.add(item)

        if item in job:
            job_education.add(item)

    if not job_education:
        return 100

    if resume_education.intersection(job_education):
        return 100

    return 40


def calculate_experience_score(resume_experience, job_text):

    job = normalize(job_text)

    required = 0

    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:year|years|yr|yrs)",
        job
    )

    if matches:

        try:
            required = max(
                float(x) for x in matches
            )
        except:
            required = 0

    if required == 0:
        return 100

    if resume_experience >= required:
        return 100

    if resume_experience > 0:
        return round(
            resume_experience / required * 100,
            2
        )

    return 20


def classify_score(score):

    if score >= 75:
        return "🟢 Strong Match"

    elif score >= 50:
        return "🟡 Moderate Match"

    return "🔴 Low Match"


def calculate_candidate(resume_text, job_text):

    resume_skills = extract_skills(resume_text)

    required_skills = extract_skills(job_text)

    matched_skills = sorted(
        set(resume_skills).intersection(
            set(required_skills)
        )
    )

    missing_skills = sorted(
        set(required_skills).difference(
            set(resume_skills)
        )
    )

    nlp_score = calculate_nlp_score(
        resume_text,
        job_text
    )

    skill_score = calculate_skill_score(
        resume_skills,
        required_skills
    )

    education_score = calculate_education_score(
        resume_text,
        job_text
    )

    experience = extract_experience(
        resume_text
    )

    experience_score = calculate_experience_score(
        experience,
        job_text
    )

    # Weighted final score
    final_score = (
        nlp_score * 0.30
        + skill_score * 0.40
        + education_score * 0.15
        + experience_score * 0.15
    )

    return {
        "nlp_score": round(nlp_score, 2),
        "skill_score": round(skill_score, 2),
        "education_score": round(education_score, 2),
        "experience_score": round(experience_score, 2),
        "final_score": round(final_score, 2),
        "resume_skills": resume_skills,
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "experience": experience
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Resume Screening & Job Matching System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered resume screening using NLP, TF-IDF, cosine similarity and skill matching.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# JOB SELECTION
# ============================================================

st.header("💼 Job Selection")

job_col1, job_col2 = st.columns([1, 2])

with job_col1:

    selected_role = st.selectbox(
        "Select Job Role",
        list(JOB_ROLES.keys()) + ["Custom Job"]
    )

with job_col2:

    if selected_role != "Custom Job":

        job_description = st.text_area(
            "Job Description",
            value=JOB_ROLES[selected_role],
            height=220
        )

    else:

        job_description = st.text_area(
            "Enter Custom Job Description",
            height=220,
            placeholder="Enter the complete job description..."
        )


if not job_description.strip():

    st.warning("Please enter a job description.")

    st.stop()


required_skills_preview = extract_skills(
    job_description
)

st.subheader("🎯 Required Skills")

if required_skills_preview:

    st.info(
        " • ".join(required_skills_preview)
    )

else:

    st.warning(
        "No recognized skills were found in the job description."
    )


st.divider()


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header("📄 Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload one or more PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)


if not uploaded_files:

    st.info(
        "Upload your candidate PDF resumes to start screening."
    )

    st.stop()


st.success(
    f"✅ {len(uploaded_files)} resume(s) uploaded."
)


# ============================================================
# PROCESS RESUMES
# ============================================================

results = []

progress = st.progress(0)

for index, uploaded_file in enumerate(uploaded_files):

    resume_text = extract_pdf_text(
        uploaded_file
    )

    if not resume_text.strip():
        continue

    candidate_name = extract_name(
        resume_text,
        uploaded_file.name
    )

    email = extract_email(
        resume_text
    )

    phone = extract_phone(
        resume_text
    )

    education = extract_education(
        resume_text
    )

    analysis = calculate_candidate(
        resume_text,
        job_description
    )

    results.append({

        "Candidate": candidate_name,

        "Resume": uploaded_file.name,

        "Email": email,

        "Phone": phone,

        "Education": ", ".join(
            education
        ) if education else "Not detected",

        "Experience": analysis["experience"],

        "NLP Score": analysis["nlp_score"],

        "Skill Score": analysis["skill_score"],

        "Education Score": analysis["education_score"],

        "Experience Score": analysis["experience_score"],

        "Final Score": analysis["final_score"],

        "Match Level": classify_score(
            analysis["final_score"]
        ),

        "Matched Skills": ", ".join(
            analysis["matched_skills"]
        ),

        "Missing Skills": ", ".join(
            analysis["missing_skills"]
        ),

        "All Skills": ", ".join(
            analysis["resume_skills"]
        ),

        "_analysis": analysis,

        "_resume_text": resume_text
    })

    progress.progress(
        (index + 1) / len(uploaded_files)
    )


progress.empty()


if not results:

    st.error(
        "No readable resumes were found."
    )

    st.stop()


# ============================================================
# DATAFRAME
# ============================================================

results.sort(
    key=lambda x: x["Final Score"],
    reverse=True
)

df = pd.DataFrame(results)


# ============================================================
# DASHBOARD
# ============================================================

st.divider()

st.header("📊 Screening Dashboard")

strong_count = sum(
    r["Final Score"] >= 75
    for r in results
)

moderate_count = sum(
    50 <= r["Final Score"] < 75
    for r in results
)

low_count = sum(
    r["Final Score"] < 50
    for r in results
)

best_score = results[0]["Final Score"]

average_score = round(
    df["Final Score"].mean(),
    2
)


c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "👥 Candidates",
        len(results)
    )

with c2:
    st.metric(
        "🏆 Best Score",
        f"{best_score}%"
    )

with c3:
    st.metric(
        "📊 Average Score",
        f"{average_score}%"
    )

with c4:
    st.metric(
        "🟢 Strong Matches",
        strong_count
    )

with c5:
    st.metric(
        "🎯 Required Skills",
        len(required_skills_preview)
    )


# ============================================================
# FILTER / SEARCH
# ============================================================

st.divider()

st.header("🔎 Search & Filter Candidates")

filter_col1, filter_col2 = st.columns(2)

with filter_col1:

    search = st.text_input(
        "Search candidate",
        placeholder="Enter candidate name..."
    )

with filter_col2:

    match_filter = st.selectbox(
        "Filter by Match Level",
        [
            "All",
            "🟢 Strong Match",
            "🟡 Moderate Match",
            "🔴 Low Match"
        ]
    )


filtered = df.copy()

if search:

    filtered = filtered[
        filtered["Candidate"]
        .str.contains(
            search,
            case=False,
            na=False
        )
    ]


if match_filter != "All":

    filtered = filtered[
        filtered["Match Level"]
        == match_filter
    ]


st.dataframe(
    filtered[
        [
            "Candidate",
            "Final Score",
            "Match Level",
            "NLP Score",
            "Skill Score",
            "Education Score",
            "Experience Score"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TOP RECOMMENDED CANDIDATE
# ============================================================

best = results[0]

st.divider()

st.header("🏆 Recommended Candidate")

st.success(
    f"{best['Candidate']} "
    f"({best['Resume']}) has the highest match score: "
    f"{best['Final Score']}%"
)


# ============================================================
# CANDIDATE DETAILS
# ============================================================

st.divider()

st.header("👥 Candidate Ranking")

for rank, result in enumerate(results, start=1):

    with st.expander(
        f"#{rank} — {result['Candidate']} — "
        f"{result['Final Score']}% — "
        f"{result['Match Level']}"
    ):

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🎯 Final Match",
                f"{result['Final Score']}%"
            )

        with col2:
            st.metric(
                "🧠 NLP",
                f"{result['NLP Score']}%"
            )

        with col3:
            st.metric(
                "🛠️ Skills",
                f"{result['Skill Score']}%"
            )

        with col4:
            st.metric(
                "💼 Experience",
                f"{result['Experience Score']}%"
            )


        info1, info2 = st.columns(2)

        with info1:

            st.subheader("👤 Candidate Information")

            st.write(
                f"**Resume:** {result['Resume']}"
            )

            st.write(
                f"**Email:** {result['Email']}"
            )

            st.write(
                f"**Phone:** {result['Phone']}"
            )

            st.write(
                f"**Education:** {result['Education']}"
            )

            st.write(
                f"**Experience:** {result['Experience']} year(s)"
            )


        with info2:

            st.subheader("🧠 Skills Analysis")

            if result["Matched Skills"]:

                st.markdown(
                    '<div class="skill-match">'
                    '<b>✅ Matched Skills</b><br><br>'
                    + result["Matched Skills"]
                    + '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.warning(
                    "No required skills matched."
                )


            st.write("")


            if result["Missing Skills"]:

                st.markdown(
                    '<div class="skill-missing">'
                    '<b>⚠️ Missing Skills</b><br><br>'
                    + result["Missing Skills"]
                    + '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.success(
                    "🎉 Candidate has all recognized required skills!"
                )


# ============================================================
# CANDIDATE COMPARISON CHART
# ============================================================

st.divider()

st.header("📈 Candidate Match Comparison")

chart_df = df[
    ["Candidate", "Final Score"]
].copy()

chart_df = chart_df.set_index(
    "Candidate"
)

st.bar_chart(
    chart_df
)


# ============================================================
# SCORE BREAKDOWN
# ============================================================

st.header("📊 Score Breakdown")

breakdown_df = df[
    [
        "Candidate",
        "NLP Score",
        "Skill Score",
        "Education Score",
        "Experience Score"
    ]
].copy()

breakdown_df = breakdown_df.set_index(
    "Candidate"
)

st.bar_chart(
    breakdown_df
)


# ============================================================
# AUTOMATIC JOB RECOMMENDATION
# ============================================================

st.divider()

st.header("🤖 Automatic Job Recommendation")

job_recommendations = []

for result in results:

    resume_text = result["_resume_text"]

    role_scores = {}

    for role, description in JOB_ROLES.items():

        role_analysis = calculate_candidate(
            resume_text,
            description
        )

        role_scores[role] = role_analysis[
            "final_score"
        ]

    best_role = max(
        role_scores,
        key=role_scores.get
    )

    job_recommendations.append({

        "Candidate": result["Candidate"],

        "Recommended Job": best_role,

        "Job Match Score": role_scores[best_role],

        "Match Level": classify_score(
            role_scores[best_role]
        )
    })


recommendation_df = pd.DataFrame(
    job_recommendations
)

recommendation_df = recommendation_df.sort_values(
    "Job Match Score",
    ascending=False
)

st.dataframe(
    recommendation_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD CSV
# ============================================================

st.divider()

st.header("📥 Download Results")

download_df = df.drop(
    columns=[
        "_analysis",
        "_resume_text"
    ],
    errors="ignore"
)

csv_data = download_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="📄 Download CSV Results",
    data=csv_data,
    file_name="AI_Resume_Screening_Results.csv",
    mime="text/csv"
)


# ============================================================
# EXCEL DOWNLOAD
# ============================================================

try:

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        download_df.to_excel(
            writer,
            index=False,
            sheet_name="Screening Results"
        )

        recommendation_df.to_excel(
            writer,
            index=False,
            sheet_name="Job Recommendations"
        )

    excel_buffer.seek(0)

    st.download_button(
        label="📊 Download Excel Report",
        data=excel_buffer,
        file_name="AI_Resume_Screening_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception:

    st.info(
        "Excel export requires openpyxl. "
        "CSV download is available above."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Resume Screening & Job Matching System | "
    "Python • NLP • TF-IDF • Cosine Similarity • Machine Learning • Streamlit"
)
