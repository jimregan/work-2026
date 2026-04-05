# Define the phonetic transformation steps
modifications = \[
    # Step 1: Normal text
    \[("find him", "/faɪnd hɪm/"),
     ("around this", "/əɹaʊnd ðɪs/"),
     ("let me", "/lɛt mi:/")\],
    # Step 2: Highlight letters that will change
    \[("fin<span class='hl_red'>d h</span>im", "/faɪnd hɪm/"),
     ("aroun<span class='hl_red'>d th</span>is", "/əɹaʊnd ðɪs/"),
     ("le<span class='hl_red'>t</span> me", "/lɛt mi:/")\],
    # Step 3: Highlight phonemes to be deleted
    \[("fin<span class='hl_red'>d h</span>im", "/faɪn<span class='hl_red'>d h</span>ɪm/"),
     ("aroun<span class='hl_red'>d th</span>is", "/əɹaʊn<span class='hl_red'>d ð</span>ɪs/"),
     ("le<span class='hl_red'>t</span> me", "/lɛ<span class='hl_red'>t</span> mi:/")\],
    # Step 4: Remove phonemes
    \[("fin<span class='hl_red'>d h</span>im", "/faɪn ɪm/"),
     ("aroun<span class='hl_red'>d th</span>is", "/əɹaʊn ɪs/"),
     ("le<span class='hl_red'>t</span> me", "/lɛ mi:/")\],
    # Step 5: Slide phonemes together
    \[("fin<span class='hl_red'>d h</span>im", "/faɪnɪm/"),
     ("aroun<span class='hl_red'>d th</span>is", "/əɹaʊnɪs/"),
     ("le<span class='hl_red'>t</span> me", "/lɛmi:/")\],
    # Step 6: More accurate phonetic representation
    \[("fin<span class='hl_red'>d h</span>im", "/faɪɾ̃ɪm/"),
     ("aroun<span class='hl_red'>d th</span>is", "/əɹaʊɾ̃ɪs/"),
     ("le<span class='hl_red'>t</span> me", "/lɛmi:/")\]
\]