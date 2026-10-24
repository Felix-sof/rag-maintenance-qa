from grounding import unsupported_numbers

HDF = (
    "failure_modes.md > Arızalar > HDF (öncelik 2, güncelleme 2026-10-08)\n"
    "Fark 8,6 K'nin altına düşer ve dönüş hızı 1380 rpm'nin altındadır. Her iki koşul birlikte sağlanmalıdır."
)
OSF = "failure_modes.md > Arızalar > OSF (öncelik 2)\n- L tipi: 11.000 minNm\n- H tipi: 13.000 minNm"


class TestSupported:
    def test_numbers_from_the_cited_passage(self):
        assert unsupported_numbers("Fark 8,6 K'nin, hız 1380 rpm'nin altında olmalıdır [1].", [HDF]) == []

    def test_decimal_comma_and_point_are_the_same_value(self):
        assert unsupported_numbers("The difference is below 8.6 K [1].", [HDF]) == []

    def test_thousands_grouping_in_any_style(self):
        for written in ("13.000", "13,000", "13 000", "13000"):
            assert unsupported_numbers(f"Sınır {written} minNm'dir [1].", [OSF]) == [], written

    def test_citation_markers_and_list_numbering_are_not_numbers(self):
        answer = "İki koşul gerekir [1, 2]:\n\n1. Fark 8,6 K altında [1].\n2. Hız 1380 rpm altında [1].\n- 3) Not."
        assert unsupported_numbers(answer, [HDF]) == []

    def test_passage_header_counts(self):
        # conflict answers quote the priority and the revision date from the header
        assert unsupported_numbers("Öncelik 2 ve 2026-10-08 tarihli [1] geçerlidir.", [HDF]) == []

    def test_numbers_from_the_question(self):
        answer = "Hayır, 1500 rpm değil; sınır 1380 rpm'dir [1]."
        assert unsupported_numbers(answer, [HDF], question="Sınır 1500 rpm mi?") == []

    def test_number_words_support_digits(self):
        assert unsupported_numbers("Bu 2 koşul birlikte sağlanmalıdır [1].", [HDF]) == []

    def test_a_percentage_matches_the_fraction_it_stands_for(self):
        tool = '{"similar": {"compared": 25, "failed": 5, "failure_rate": 0.2}}'
        assert unsupported_numbers("25 prosesin 5'i (%20) arızalandı [T1].", [tool]) == []
        assert unsupported_numbers("Oran 20 % [T1].", [tool]) == []
        assert unsupported_numbers("Oran %0,1 [1].", ["RNF %0,1 olasılıkla oluşur."]) == []
        assert unsupported_numbers("Olasılık 0.001 [1].", ["RNF %0,1 olasılıkla oluşur."]) == []

    def test_a_wrong_percentage_is_still_flagged(self):
        assert unsupported_numbers("Oran %30 [T1].", ['{"failure_rate": 0.2}']) == ["30"]
        assert unsupported_numbers("20 proses [T1].", ['{"failure_rate": 0.2}']) == ["20"]  # no % sign: not 0.2

    def test_digits_inside_identifiers_are_ignored(self):
        assert unsupported_numbers("AI4I veri setinde H2O yoktur [1].", [HDF]) == []


class TestUnsupported:
    def test_a_number_no_cited_passage_contains(self):
        assert unsupported_numbers("Hız 1500 rpm'nin altında olmalıdır [1].", [HDF]) == ["1500"]

    def test_a_number_from_an_uncited_passage(self):
        # 13.000 is in the knowledge base, but not in the passage the answer cites
        assert unsupported_numbers("Sınır 13.000 minNm'dir [1].", [HDF]) == ["13.000"]

    def test_a_changed_decimal(self):
        assert unsupported_numbers("Fark 8,2 K'nin altına düşer [1].", [HDF]) == ["8,2"]

    def test_a_round_thousand_is_not_supported_by_a_small_integer(self):
        # HDF's header carries the date 2026-10-08; that "10" must not vouch for 10.000
        assert unsupported_numbers("Sınır 10.000 minNm'dir [1].", [HDF]) == ["10.000"]

    def test_parts_of_a_decimal_do_not_support_other_numbers(self):
        assert unsupported_numbers("6 dakika bekleyin [1].", [HDF]) == ["6"]

    def test_reported_once_in_order_as_written(self):
        answer = "Önce 45 dakika, sonra 1.500 rpm, yine 45 dakika [1]."
        assert unsupported_numbers(answer, [HDF]) == ["45", "1.500"]

    def test_nothing_cited_supports_nothing(self):
        assert unsupported_numbers("Hız 1380 rpm'nin altındadır.", []) == ["1380"]


def test_dates_and_versions_are_checked_group_by_group():
    passage = "kilavuz.pdf > s. 3 (öncelik 2)\nRevizyon 08.10.2026, yazılım 1.2.3."
    assert unsupported_numbers("08.10.2026 tarihli revizyon, sürüm 1.2.3 [1].", [passage]) == []
    assert unsupported_numbers("09.10.2026 tarihli revizyon [1].", [passage]) == ["09.10.2026"]
