"""PII/secret detection and masking (PRIVACY; FR-2, NFR-4, AC-6)."""

import privacy
from privacy import contains_pii, mask, restore, scan


def test_email_detected_and_masked():
    result = mask("write to ada@example.com please")
    assert result.text == "write to [EMAIL_1] please"
    assert result.mapping == {"[EMAIL_1]": "ada@example.com"}
    assert result.counts == {"EMAIL": 1}


def test_secret_line_labeled_secret():
    findings = scan("api_key=sk-testvalue123")
    assert findings and findings[0].category == "SECRET"


def test_card_needs_luhn():
    assert contains_pii("card 4111111111111111")
    assert not contains_pii("card 4111111111111112")


def test_phone_ip_govid_detected():
    assert contains_pii("call +1-415-555-0132")
    assert contains_pii("host 10.0.0.8 went down")
    assert contains_pii("id 123-45-6789 on file")


def test_clean_text_untouched():
    assert mask("remember that I like concise answers").text == (
        "remember that I like concise answers"
    )
    assert not contains_pii("remember that I like concise answers")


def test_counters_increment_per_category():
    result = mask("a@b.co and c@d.co")
    assert result.text == "[EMAIL_1] and [EMAIL_2]"


def test_originals_never_appear_in_masked_text():
    result = mask("mail ada@example.com, pass password=hunter2!")
    for original in result.mapping.values():
        assert original not in result.text


def test_restore_is_opt_in():
    result = mask("mail ada@example.com")
    assert restore(result.text, result.mapping) == "mail ada@example.com"
