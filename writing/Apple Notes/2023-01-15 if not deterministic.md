if not deterministic:
            # This includes alternate vocalization (hour menos min, min para hour), here we shift the times and indicate a `style` tag
            hour_shift_24 = pynini.invert(pynini.string_file(get_abs_path("data/time/hour_to_24.tsv")))
            hour_shift_12 = pynini.invert(pynini.string_file(get_abs_path("data/time/hour_to_12.tsv")))
            minute_shift = pynini.string_file(get_abs_path("data/time/minute_to.tsv"))

            graph_hour_to_24 = graph_24 @ hour_shift_24 @ cardinal_graph
            graph_hour_to_12 = graph_12 @ hour_shift_12 @ cardinal_graph

            graph_minute_to = pynini.union(graph_minute_single, graph_minute_double) @ minute_shift @ cardinal_graph

            final_graph_hour_to_24 = pynutil.insert("hours: \"") + graph_hour_to_24 + pynutil.insert("\"")
            final_graph_hour_to_12 = pynutil.insert("hours: \"") + graph_hour_to_12 + pynutil.insert("\"")

            final_graph_minute_to = pynutil.insert("minutes: \"") + graph_minute_to + pynutil.insert("\"")

            graph_menos = pynutil.insert(" style: \"1\"")
            graph_para = pynutil.insert(" style: \"2\"")

            final_graph_style = graph_menos | graph_para