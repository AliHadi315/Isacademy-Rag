"""Bilingual UI strings (English / Arabic) + RTL styling.

Every user-facing string in the app comes from here, so adding a language is
one column in one dict. `t("key")` returns the string for the active language.

Arabic is right-to-left, so `rtl_css()` flips text direction for the page
chrome while deliberately leaving code blocks, dataframes and Plotly charts
alone - those are LTR by nature and flipping them looks broken.
"""
from __future__ import annotations

LANGUAGES = {"en": "English", "ar": "العربية"}

STRINGS: dict[str, dict[str, str]] = {
    # --- chrome ---
    "app_title": {"en": "Isacademy AI", "ar": "إسأكاديمي AI"},
    "app_subtitle": {
        "en": "Document Intelligence with Gemini",
        "ar": "تحليل المستندات باستخدام Gemini",
    },
    "language": {"en": "Language", "ar": "اللغة"},
    "nav_dashboard": {"en": "Dashboard", "ar": "لوحة التحكم"},
    "nav_documents": {"en": "Documents", "ar": "المستندات"},
    "nav_ask": {"en": "Ask Questions", "ar": "اطرح سؤالاً"},
    "nav_data": {"en": "Data Analysis", "ar": "تحليل البيانات"},
    "nav_reports": {"en": "Reports", "ar": "التقارير"},
    "nav_settings": {"en": "Settings", "ar": "الإعدادات"},

    # --- dashboard ---
    "documents": {"en": "Documents", "ar": "المستندات"},
    "chunks": {"en": "Chunks", "ar": "المقاطع"},
    "questions_answered": {"en": "Questions Answered", "ar": "الأسئلة المُجابة"},
    "quick_actions": {"en": "Quick Actions", "ar": "إجراءات سريعة"},
    "upload_document": {"en": "Upload Document", "ar": "رفع مستند"},
    "ask_a_question": {"en": "Ask a Question", "ar": "اطرح سؤالاً"},
    "analyze_data": {"en": "Analyze Data", "ar": "تحليل البيانات"},

    # --- documents ---
    "upload_pdfs": {"en": "Upload PDF documents", "ar": "ارفع مستندات PDF"},
    "process_files": {"en": "Process files", "ar": "معالجة الملفات"},
    "force_reindex": {"en": "Force re-index", "ar": "إعادة الفهرسة إجبارياً"},
    "indexed_documents": {"en": "Indexed documents", "ar": "المستندات المفهرسة"},
    "pages": {"en": "pages", "ar": "صفحات"},
    "figures": {"en": "figures", "ar": "أشكال"},
    "tables": {"en": "tables", "ar": "جداول"},
    "delete": {"en": "Delete", "ar": "حذف"},
    "reindex": {"en": "Re-index", "ar": "إعادة الفهرسة"},
    "no_documents": {
        "en": "No documents indexed yet. Upload a PDF to begin.",
        "ar": "لا توجد مستندات مفهرسة بعد. ارفع ملف PDF للبدء.",
    },
    "already_indexed": {
        "en": "Already indexed (identical file) - skipped.",
        "ar": "تمت فهرسته مسبقاً (ملف مطابق) - تم التخطي.",
    },
    "processing_done": {"en": "Ready", "ar": "جاهز"},
    "load_demo": {"en": "Load demo documents", "ar": "تحميل مستندات تجريبية"},

    # --- ask ---
    "ask_title": {"en": "Ask a Question", "ar": "اطرح سؤالاً"},
    "ask_placeholder": {
        "en": "Enter your question...",
        "ar": "اكتب سؤالك...",
    },
    "ask_button": {"en": "Ask", "ar": "اسأل"},
    "answer": {"en": "Answer", "ar": "الإجابة"},
    "sources": {"en": "Sources", "ar": "المصادر"},
    "page": {"en": "Page", "ar": "الصفحة"},
    "relevant_visual": {"en": "Relevant Visual", "ar": "العنصر المرئي ذو الصلة"},
    "visual_analysis": {"en": "Visual Analysis", "ar": "تحليل العنصر المرئي"},
    "what_visual_shows": {"en": "What the visual shows", "ar": "ما يعرضه العنصر المرئي"},
    "key_findings": {"en": "Key Findings", "ar": "أهم النتائج"},
    "source": {"en": "Source", "ar": "المصدر"},
    "thinking": {"en": "Working...", "ar": "جاري العمل..."},
    "example_questions": {"en": "Example questions", "ar": "أسئلة نموذجية"},

    # --- PCA ---
    "semantic_pca_map": {"en": "Semantic PCA Map", "ar": "خريطة PCA الدلالية"},
    "query": {"en": "Query", "ar": "السؤال"},
    "cluster": {"en": "Cluster", "ar": "المجموعة"},
    "query_cluster": {"en": "Query Cluster", "ar": "مجموعة السؤال"},
    "closest_items": {"en": "Closest Items", "ar": "العناصر الأقرب"},
    "closest_to_question": {
        "en": "Closest to your question",
        "ar": "الأقرب إلى سؤالك",
    },
    "similarity": {"en": "Similarity", "ar": "درجة التشابه"},
    "why_close": {"en": "Why Are They Close?", "ar": "لماذا هذه العناصر قريبة؟"},
    "nearby_cluster": {"en": "Nearby cluster", "ar": "مجموعة قريبة"},
    "closest_cluster": {"en": "Closest cluster", "ar": "المجموعة الأقرب"},
    "other_nearby": {"en": "Other nearby items", "ar": "عناصر قريبة أخرى"},
    "pca_limitation": {
        "en": ("PCA is a 2D projection of the embedding space. Visual proximity is an "
               "approximation and should be interpreted together with similarity scores "
               "and sources."),
        "ar": ("تحليل PCA هو إسقاط ثنائي الأبعاد لمساحة التضمين. القرب البصري هو تقريب "
               "ويجب تفسيره مع درجات التشابه والمصادر."),
    },
    "pca_unavailable": {
        "en": ("PCA is unavailable because there are not enough relevant embedding "
               "vectors for a meaningful visualization."),
        "ar": ("تحليل PCA غير متاح لعدم وجود عدد كافٍ من المتجهات لإنشاء تصور مرئي موثوق."),
    },

    # --- data analysis ---
    "data_title": {"en": "Data Analysis", "ar": "تحليل البيانات"},
    "upload_dataset": {"en": "Upload a dataset (CSV / Excel)", "ar": "ارفع مجموعة بيانات (CSV / Excel)"},
    "dataset": {"en": "Dataset", "ar": "مجموعة البيانات"},
    "rows": {"en": "Rows", "ar": "الصفوف"},
    "columns": {"en": "Columns", "ar": "الأعمدة"},
    "numeric": {"en": "Numeric", "ar": "رقمية"},
    "missing": {"en": "Missing", "ar": "قيم مفقودة"},
    "features": {"en": "Features", "ar": "الخصائص"},
    "run_pca": {"en": "Run PCA", "ar": "تشغيل PCA"},
    "explained_variance": {"en": "Explained variance", "ar": "التباين المُفسَّر"},
    "preview": {"en": "Preview", "ar": "معاينة"},
    "charts": {"en": "Charts", "ar": "الرسوم البيانية"},
    "generate_chart": {"en": "Generate chart", "ar": "إنشاء الرسم"},
    "no_dataset": {
        "en": "No dataset yet. Upload a CSV or Excel file above.",
        "ar": "لا توجد بيانات بعد. ارفع ملف CSV أو Excel أعلاه.",
    },
    "no_numeric_columns": {
        "en": "This dataset has no numeric columns to analyse.",
        "ar": "لا تحتوي هذه البيانات على أعمدة رقمية للتحليل.",
    },

    # --- reports ---
    "reports_title": {"en": "Reports", "ar": "التقارير"},
    "generate_report": {"en": "Generate report", "ar": "إنشاء تقرير"},
    "report_title_label": {"en": "Report title", "ar": "عنوان التقرير"},
    "formats": {"en": "Formats", "ar": "الصيغ"},
    "download": {"en": "Download", "ar": "تحميل"},
    "previous_reports": {"en": "Previously generated reports", "ar": "التقارير السابقة"},
    "no_report_source": {
        "en": "Ask a question first - a report is built from a completed answer.",
        "ar": "اطرح سؤالاً أولاً - يُبنى التقرير من إجابة مكتملة.",
    },
    "question": {"en": "Question", "ar": "السؤال"},
    "cluster_explanation": {"en": "Cluster Explanation", "ar": "شرح المجموعة"},

    # --- settings ---
    "settings_title": {"en": "Settings", "ar": "الإعدادات"},
    "gemini_model": {"en": "Gemini model", "ar": "نموذج Gemini"},
    "top_k": {"en": "Results per search (Top K)", "ar": "عدد النتائج (Top K)"},
    "pca_max_points": {"en": "PCA max points", "ar": "أقصى عدد نقاط في PCA"},
    "apply": {"en": "Apply", "ar": "تطبيق"},
    "applied": {"en": "Settings applied for this session.", "ar": "تم تطبيق الإعدادات لهذه الجلسة."},
    "current_config": {"en": "Current configuration", "ar": "الإعدادات الحالية"},

    # --- errors ---
    "no_api_key": {
        "en": ("No Gemini API key configured. Add GEMINI_API_KEY to your .env file "
               "and restart the app."),
        "ar": ("لم يتم ضبط مفتاح Gemini. أضف GEMINI_API_KEY إلى ملف .env ثم أعد تشغيل التطبيق."),
    },
    "gemini_failed": {
        "en": "Gemini could not be reached",
        "ar": "تعذّر الوصول إلى Gemini",
    },
    "no_evidence": {
        "en": ("I could not find enough information in the uploaded documents to answer "
               "this question reliably."),
        "ar": ("لم أجد معلومات كافية في المستندات المرفوعة للإجابة عن هذا السؤال بشكل موثوق."),
    },
    "empty_index": {
        "en": "No documents are indexed yet. Upload a PDF on the Documents page.",
        "ar": "لا توجد مستندات مفهرسة. ارفع ملف PDF من صفحة المستندات.",
    },
    "no_visual_found": {
        "en": "No visual could be located for this question.",
        "ar": "تعذّر العثور على عنصر مرئي لهذا السؤال.",
    },
    "extraction_failed": {"en": "Could not read this file", "ar": "تعذّرت قراءة هذا الملف"},
}


def t(key: str, lang: str = "en") -> str:
    """Translate a key. Falls back to English, then to the key itself."""
    entry = STRINGS.get(key)
    if not entry:
        return key
    return entry.get(lang) or entry.get("en") or key


def is_rtl(lang: str) -> bool:
    return lang == "ar"


RTL_CSS = """
<style>
  /* Arabic: flip the page chrome, but never the things that read LTR by
     nature - code, dataframes and Plotly charts stay as they are. */
  section.main .block-container, [data-testid="stSidebar"] {
      direction: rtl;
      text-align: right;
  }
  section.main h1, section.main h2, section.main h3,
  section.main p, section.main li, section.main label {
      text-align: right;
  }
  [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { direction: ltr; }
  .stCode, pre, code, [data-testid="stDataFrame"], .js-plotly-plot {
      direction: ltr;
      text-align: left;
  }
  .isa-src, .isa-ltr { direction: ltr; text-align: left; unicode-bidi: isolate; }
</style>
"""
