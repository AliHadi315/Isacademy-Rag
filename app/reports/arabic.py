"""Arabic text support for PDF output.

ReportLab's built-in fonts (Helvetica, Times) contain no Arabic glyphs, so
Arabic text is silently dropped from a generated PDF. Two things are needed to
fix that:

1. **A font that has the glyphs.** `register_arabic_font()` finds a suitable
   TrueType font on the system and registers it with ReportLab.
2. **Shaping and direction.** Arabic letters change form depending on their
   neighbours, and the script runs right to left. `shape()` applies
   `arabic-reshaper` for the letter forms and `python-bidi` for the direction.

Both dependencies are optional: without them the module degrades to plain
Latin rendering rather than failing, and `arabic_ready()` reports which state
the app is in so the UI can be honest about it.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.utils.logging import get_logger

log = get_logger("arabic")

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

# Fonts that ship with common systems and cover Arabic.
FONT_CANDIDATES = [
    ("Arial", r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("Tahoma", r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\tahomabd.ttf"),
    ("SegoeUI", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
    ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("NotoSansArabic", "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf", ""),
    ("ArialUnicode", "/Library/Fonts/Arial Unicode.ttf", ""),
]

_registered: str | None = None
_checked = False


def has_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(str(text or "")))


def register_arabic_font() -> str | None:
    """Register an Arabic-capable font. Returns its name, or None."""
    global _registered, _checked
    if _checked:
        return _registered
    _checked = True

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.fonts import addMapping
    except ImportError:
        return None

    for name, regular, bold in FONT_CANDIDATES:
        if not regular or not Path(regular).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, regular))
            if bold and Path(bold).exists():
                pdfmetrics.registerFont(TTFont(name + "-Bold", bold))
                addMapping(name, 0, 0, name)
                addMapping(name, 1, 0, name + "-Bold")
            _registered = name
            log.info("Arabic font registered: %s", name)
            return name
        except Exception as exc:
            log.debug("could not register %s: %s", name, exc)

    log.warning("no Arabic-capable font found - Arabic will not render in PDFs")
    return None


def shape(text: str) -> str:
    """Join Arabic letters and apply right-to-left ordering.

    Latin text passes through untouched, so it is safe to call on anything.
    """
    text = str(text or "")
    if not has_arabic(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        log.warning(
            "arabic-reshaper / python-bidi are missing - Arabic will render "
            "unjoined. Install them with: pip install arabic-reshaper python-bidi"
        )
        return text
    except Exception as exc:
        log.warning("Arabic shaping failed (%s) - using the raw text", exc)
        return text


def arabic_ready() -> tuple[bool, str]:
    """(is Arabic PDF output fully supported, explanation)."""
    font = register_arabic_font()
    try:
        import arabic_reshaper  # noqa: F401
        from bidi.algorithm import get_display  # noqa: F401

        shaping = True
    except ImportError:
        shaping = False

    if font and shaping:
        return True, "Arabic PDF export is available (font: " + font + ")."
    if not font and not shaping:
        return False, ("Arabic PDF export needs an Arabic font and "
                       "arabic-reshaper + python-bidi.")
    if not font:
        return False, "No Arabic-capable TrueType font was found on this system."
    return False, ("Install arabic-reshaper and python-bidi for correct Arabic "
                   "letter joining in PDFs.")
