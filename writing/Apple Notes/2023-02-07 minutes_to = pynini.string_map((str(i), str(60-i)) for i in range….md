minutes_to = pynini.string_map(\[(str(i), str(60-i)) for i in range(1, 60)\])
        minutes_inverse = pynini.invert(
            pynini.project(minutes_to, "input")
            @ tn_cardinal_tagger.graph_en
        )
        minute_words_to_words = minutes_inverse @ minutes_to @ tn_cardinal_tagger.graph_en
        minute_words_to_words = pynutil.insert("minutes: \"") + minute_words_to_words + pynutil.insert("\"")
        def hours_to_pairs():
            for x in range(1, 13):
                if x == 12:
                    y = 1
                else:
                    y = x + 1
                yield x, y
        hours_to = pynini.string_map(\[(str(x\[0\]), str(x\[1\])) for x in hours_to_pairs()\])