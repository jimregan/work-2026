graph_unit = pynini.string_file(get_abs_path("data/measure/unit.tsv"))
        graph_unit_ett = pynini.string_file(get_abs_path("data/measure/unit_neuter.tsv"))
        graph_plurals = pynini.string_file(get_abs_path("data/measure/unit_plural.tsv"))

        graph_unit |= pynini.compose(
            pynini.closure(TO_LOWER, 1) + (SV_ALPHA | TO_LOWER) + pynini.closure(SV_ALPHA | TO_LOWER), graph_unit
        ).optimize()
        graph_unit_ett |= pynini.compose(
            pynini.closure(TO_LOWER, 1) + (SV_ALPHA | TO_LOWER) + pynini.closure(SV_ALPHA | TO_LOWER), graph_unit_ett
        ).optimize()

        graph_unit_plural = convert_space(graph_unit @ graph_plurals)
        graph_unit_plural_ett = convert_space(graph_unit_ett @ graph_plurals)
        graph_unit = convert_space(graph_unit)
        graph_unit_ett = convert_space(graph_unit_ett)


cardinal_graph_hundreds_one_non_zero