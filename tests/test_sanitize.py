"""Tests for web content sanitization."""

from hydrag import sanitize_web_content


class TestSanitizeWebContent:
    def test_strips_script_tags(self) -> None:
        raw = "Before<script>alert('xss')</script>After"
        assert "alert" not in sanitize_web_content(raw)
        assert "Before" in sanitize_web_content(raw)
        assert "After" in sanitize_web_content(raw)

    def test_strips_style_tags(self) -> None:
        raw = "Content<style>.foo{color:red}</style>More"
        result = sanitize_web_content(raw)
        assert "color:red" not in result
        assert "Content" in result

    def test_strips_html_tags(self) -> None:
        raw = "<div><p>Hello <b>world</b></p></div>"
        result = sanitize_web_content(raw)
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_unescapes_html_entities(self) -> None:
        raw = "foo &amp; bar &lt;baz&gt;"
        result = sanitize_web_content(raw)
        assert "foo & bar <baz>" == result.strip()

    def test_strips_markdown_by_default(self) -> None:
        raw = "# Heading\n**bold** text [link](http://example.com)"
        result = sanitize_web_content(raw)
        assert "#" not in result
        assert "**" not in result
        assert "http://example.com" not in result
        assert "link" in result
        assert "bold" in result

    def test_preserves_markdown_when_allowed(self) -> None:
        raw = "# Heading\n**bold** [link](http://example.com)"
        result = sanitize_web_content(raw, allow_markdown=True)
        assert "# Heading" in result
        assert "**bold**" in result

    def test_strips_image_markdown(self) -> None:
        raw = "See ![alt text](http://example.com/img.png) here"
        result = sanitize_web_content(raw)
        assert "http://example.com" not in result
        assert "alt text" in result

    def test_max_chars_truncation(self) -> None:
        raw = "a" * 5000
        result = sanitize_web_content(raw, max_chars=100)
        assert len(result) == 100

    def test_collapses_excessive_newlines(self) -> None:
        raw = "line1\n\n\n\n\nline2"
        result = sanitize_web_content(raw)
        assert "\n\n\n" not in result
        assert "line1\n\nline2" == result

    def test_empty_input(self) -> None:
        assert sanitize_web_content("") == ""

    def test_combined_html_and_markdown(self) -> None:
        raw = "<div>## Title</div>\n<p>**bold** and <a href='#'>link</a></p>"
        result = sanitize_web_content(raw)
        assert "<" not in result
        assert "**" not in result
        assert "Title" in result
        assert "bold" in result
