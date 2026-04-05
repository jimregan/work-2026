zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
        digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))

        # Any single digit
        graph_digit = digit
        digit_inverse = pynini.invert(digit)
        digits_no_one = (NEMO_DIGIT - "1") @ graph_digit

        graph_zero = zero
        if not deterministic:
            graph_zero |= pynini.cross("0", "nulla")
            graph_digit |= pynini.cross("1", "akta")

        teen = pynutil.delete("1") + digit + pynutil.insert("nuppelohkái")
        teen |= pynini.cross("10", "logi")
        ties = digits_no_one + pynini.cross("0", "logi")
        ties |= digits_no_one + pynutil.insert("logi") + digit

        graph_tens = teen
        graph_ties = ties

        self.tens = graph_tens.optimize()
        self.ties = graph_ties.optimize()

        two_digit_non_zero = pynini.union(graph_tens, graph_ties, (pynutil.delete("0") + graph_digit))
        graph_two_digit_non_zero = pynini.union(graph_digit, two_digit_non_zero)

        self.two_digit_non_zero = graph_two_digit_non_zero.optimize()

        # Three digit strings
        hundreds = digits_no_one + pynutil.insert("čuođi")
        hundreds |= pynini.cross("1", "čuođi")
        if not deterministic:
            hundreds |= pynini.cross("1", "oktačuođi")
            hundreds |= pynini.cross("1", "aktačuođi")

        final_hundreds = hundreds + pynini.union(two_digit_non_zero, pynutil.delete("00"))
        graph_hundreds = pynini.union(final_hundreds, graph_two_digit_non_zero)

        self.hundreds = graph_hundreds.optimize()

        # For all three digit strings with leading zeroes (graph appends '0's to manage place in string)
        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + (graph_tens | graph_ties))

        graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )
        graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )
        self.graph_hundreds_component_at_least_one_non_zero_digit = (
            graph_hundreds_component_at_least_one_non_zero_digit
        )
        self.graph_hundreds_component_at_least_one_non_zero_digit_no_one = (
            graph_hundreds_component_at_least_one_non_zero_digit_no_one.optimize()
        )

        duhat = pynutil.insert("duhát")
        duhat_cross = pynini.cross("001", "duhát")
        if not deterministic:
            duhat_cross |= pynini.cross("001", "duhát ")

        graph_thousands_component_at_least_one_non_zero_digit = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit,
            graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + duhat
            + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
            duhat_cross + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
        )
        self.graph_thousands_component_at_least_one_non_zero_digit = (
            graph_thousands_component_at_least_one_non_zero_digit
        )

        graph_thousands_component_at_least_one_non_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit_no_one,
            graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + duhat
            + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
            duhat_cross + (graph_hundreds_component_at_least_one_non_zero_digit | pynutil.delete("000")),
        )

        graph_million = make_million("m", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic)

        graph_milliard = make_milliard("m", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic)

        graph_billion = make_million("b", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic)

        graph_billiard = make_milliard("b", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic)

        graph_trillion = make_million("tr", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic)

        graph_trilliard = make_milliard(
            "tr", graph_hundreds_component_at_least_one_non_zero_digit_no_one, deterministic
        )

        self.graph_higher = (
            graph_trilliard + graph_trillion + graph_billiard + graph_billion + graph_milliard + graph_million
        )
        graph = self.graph_higher + (graph_thousands_component_at_least_one_non_zero_digit | pynutil.delete("000000"))

        self.graph = (
            ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "\[BOS\]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 24
            @ graph
            @ pynini.cdrewrite(delete_space, "\[BOS\]", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_space, "", "\[EOS\]", NEMO_SIGMA)
            @ pynini.cdrewrite(
                pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), SE_ALPHA, SE_ALPHA, NEMO_SIGMA
            )
        )
        self.graph |= graph_zero

        self.graph = filter_punctuation(self.graph).optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()