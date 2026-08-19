# Chat Conversation

*Exported on 19/08/2026, 11:50:42*

---

## User

Can you find me examples in the literature of multi task learning where the tasks were curated according to their expected contribution to another task? I mean with a concrete task, rather than generic few or zero shot

---

## Assistant

Yes. If by curated you mean that the auxiliary tasks were not simply "all available tasks" but were deliberately chosen (or automatically selected) because they were expected to improve a particular target task, there is a fairly distinct literature around auxiliary task selection, target-task-oriented MTL, and asymmetric transfer.
A useful distinction is:


Hand-curated auxiliary tasks chosen because of a hypothesized benefit to a specific target task.


Data-driven task selection where the system explicitly estimates which tasks help the target task and selects them.


Task-relatedness estimation where task relationships are measured and only beneficial transfers are allowed.


Some concrete examples:

1. Søgaard & Goldberg (2016): low-level tasks chosen to improve a higher-level NLP task
Target tasks:


Semantic tagging


Chunking


CCG supertagging


Auxiliary tasks:


POS tagging


Other lower-level linguistic annotations


The key idea was not "do many tasks." Rather, they investigated which linguistic tasks should be supervised at lower layers because they are expected to support a specific downstream task. The auxiliary tasks were chosen according to a linguistic hierarchy of expected usefulness. This is one of the classic examples of deliberately curated task relationships. arXiv+1
Why it's relevant:


Concrete target task.


Auxiliary tasks selected because they are believed to provide useful inductive bias for that target.



2. PAD-Net (Depth Estimation + Scene Parsing)
Target tasks:


Depth estimation


Scene parsing


Intermediate auxiliary tasks:


Surface normals


Object boundaries


Other geometric cues


PAD-Net explicitly creates intermediate predictions because those tasks are expected to contribute information useful to the final tasks. The auxiliary tasks are not arbitrary—they are selected based on the causal/geometric relationship to the target outputs. GitHub
This is one of the clearest computer vision examples of "choose task B because it should help task A."

3. Liebel & Körner (2018): Auxiliary Tasks in Multi-task Learning
Road-scene understanding target tasks:


Depth


Segmentation


Scene understanding


They systematically introduced additional tasks that were not directly required at inference time but were expected to regularize representation learning for the main tasks. Their study explicitly evaluates which auxiliary labels improve the target objectives. arXiv
What makes it interesting:


Many candidate annotations.


Auxiliary tasks chosen because of expected contribution to road-scene understanding.



4. Bjerva (2017): Estimating whether an auxiliary tagging task will help
This paper is almost exactly about your question.
Target task:


Morphosyntactic or semantic tagging


Candidate auxiliary tasks:


Other linguistic annotation tasks


The goal was to predict before training whether an auxiliary task is likely to improve the target task. They showed that measures based on joint label distributions are predictive of usefulness. ep.liu.se
This is less about building the MTL model and more about:

"How do I decide which task to include because it will help my target?"


5. AutoSeM (Guo et al., 2019)
Target:


A chosen GLUE task


Candidate auxiliary tasks:


Other GLUE tasks


AutoSeM explicitly frames auxiliary-task selection as an optimization problem. A bandit algorithm selects tasks that improve the target task, and then learns the mixing ratio. arXiv
This is one of the cleanest examples of:

Given many possible tasks, automatically curate the subset expected to improve task X.


6. AMTL (Lee et al., 2016)
Target:


Multiple tasks, but transfer is asymmetric


Instead of assuming all tasks should help each other equally, AMTL learns a directed graph of task influence. Stronger or more relevant tasks are allowed to transfer to weaker tasks while suppressing harmful transfer. Proceedings of Machine Learning Research
Not exactly task selection, but very close conceptually:


Estimate which tasks contribute to which others.


Prevent irrelevant tasks from influencing a target.



7. Auxiliary RL tasks (UNREAL / Jaderberg et al. lineage)
Target:


Policy learning


Auxiliary tasks:


Pixel control


Reward prediction


Value-function related objectives


These auxiliary tasks were specifically engineered because they were expected to improve representation learning for the control task. This became a major paradigm in RL auxiliary learning. The later literature explicitly treats auxiliary-task design as choosing tasks that maximize improvement of a designated target task. arXiv+1

8. Recent "target-task auxiliary selection" literature
A more recent line of work treats the problem directly:


Learning to Leverage Auxiliary Tasks for Multitask Learning (2024)


Efficient Multi-Task Auxiliary Learning (EMNLP 2021)


AutoSeM (2019)


These papers explicitly assume:


one target task,


many candidate auxiliary tasks,


select/weight auxiliary tasks according to their expected effect on the target task. arXiv+2ACL Anthology+2



What I'd cite if you're writing a paper
For a literature review specifically about curating tasks according to expected contribution to a concrete target task, I'd organize it as:
CategoryRepresentative papersManual task curationSøgaard & Goldberg (2016), PAD-Net (2018), Liebel & Körner (2018)Task usefulness estimationBjerva (2017)Automatic task selectionAutoSeM (2019), Efficient Multi-Task Auxiliary Learning (2021)Learned task influence graphsAMTL (2016)Meta-learned auxiliary tasksMAXL (2019), recent auxiliary-task-generation work
One thing I'd note is that surprisingly little of the classical MTL literature does principled task curation. Many papers simply bundle related tasks together (e.g., POS + NER + chunking). The papers above are the ones that explicitly ask:

"Which task should I add if my objective is to improve this particular target task?"

That's much closer to the question you're asking.

---

## Assistant

Exactly. I think "rules" undersells what you're doing.
A traditional G2P system contains at least three different kinds of knowledge:


Rules


grapheme-to-phoneme correspondences


context-sensitive rewrite rules


stress rules


vowel reduction rules




Intermediate analyses


syllabification


morphological decomposition


stress assignment


grapheme classification


onset/rhyme structure




Pipeline structure


perform morphology before pronunciation


determine syllable boundaries before stress


determine stress before vowel reduction


exception lookup before rule application




What you're doing sounds more like:

We identify the intermediate representations and processing stages that occur in classical G2P pipelines and expose them as auxiliary supervision tasks for a neural sequence model.

That's a stronger story than:

We turn rules into tasks.

Because many of the useful things aren't really rules at all.
For example:
Classical componentRule?Auxiliary task?Syllabificationnot reallyyesMorphological segmentationnot reallyyesStress assignmentpartlyyesGrapheme classificationnoyesException detectionnoyesLetter-to-sound mappingyesyes
A speech reviewer is likely to recognize these as familiar modules from systems like festival, lexicon builders, and older pronunciation-generation pipelines.

In fact, if I were writing the motivation section, I'd probably avoid talking about "rules" until later and instead frame it as:

Auxiliary tasks were selected by analysing the information flow in conventional grapheme-to-phoneme conversion systems. Tasks correspond either to intermediate representations (e.g. syllabification and morphology) or to decision processes (e.g. stress assignment) that are traditionally computed explicitly before pronunciation generation.

That sounds much more like a systems paper and less like a symbolic-vs-neural argument.

There's also a subtle distinction that may help you conceptually:
Most MTL papers ask:

What tasks are statistically related to the target task?

You're asking:

What information would a competent pronunciation engineer explicitly compute while solving the target task?

That's a different criterion.
It's closer to:


inductive bias,


structured prediction,


linguistic scaffolding,


representational decomposition,


than to generic task similarity.
Honestly, that's probably the more interesting contribution. If the auxiliary tasks are chosen because they correspond to known latent variables in pronunciation generation, then the work is not merely "MTL for G2P"; it's a claim that neural G2P benefits from supervision on the same intermediate structures that classical G2P systems were engineered to compute. That's a much more coherent narrative for an audience that still has one foot in the eSpeak/Festival world.

---

## User

A few of the tasks will involve generated data, or different views of the same data source

---

## Assistant

That actually strengthens the connection to the pipeline view, provided you're careful in how you describe it.
In multitask learning, reviewers often become suspicious when they see a long list of auxiliary tasks and discover that several are derived from the same underlying annotation. The immediate question is:

Are these genuinely different tasks, or just multiple encodings of the same label?

For your setup, I would lean into the idea that they are different representations of information used at different stages of pronunciation generation.
For example, suppose you have a pronunciation lexicon:
orthography: photographphonemes: F OW T AH G R AE F
From that you might derive:


syllable boundaries


stress pattern


vowel/consonant sequence


phoneme classes


sonority profile


onset/coda structure


These are not independent data sources, but they are distinct views of the pronunciation.
Similarly, from orthography you might derive:


morphological segmentation


affix labels


grapheme classes


vowel/consonant patterns


Again, different views of the same word.
That is fairly common in structured prediction research. The important point is not that the tasks come from different datasets but that they expose different latent structure.

I think there are three categories of auxiliary tasks you could distinguish:
1. External linguistic annotations
Examples:


morphology corpus


syllabification dictionary


stress annotations


These are easiest to justify because they are independent sources of supervision.
2. Derived linguistic representations
Examples:


phoneme classes derived from phoneme strings


onset/rime decomposition


vowel/consonant masks


These are generated automatically from existing annotations.
The justification is:

They make explicit information that is implicit in the target representation.

3. Alternative views of the same prediction problem
Examples:
word -> phonemesword -> stress patternword -> syllablesword -> phoneme classes
These are arguably not separate knowledge sources at all. They're different projections of the pronunciation.
But that is not necessarily a weakness. In fact, there is a substantial literature on auxiliary supervision through alternative label spaces.

One thing I would avoid saying is:

We selected tasks because they are different.

Instead I'd say:

We selected tasks because they correspond to distinct intermediate representations or decision points within traditional pronunciation-generation pipelines.

Then whether the labels come from:


independent resources,


generated annotations,


deterministic transformations of the target,


becomes a secondary issue.

For a T5 implementation specifically, I'd even consider organizing the task inventory according to provenance:
TypeExamplePrimary taskorthography → phonemesExternal auxiliaryorthography → morphologyDerived auxiliaryphonemes → stress patternDerived auxiliaryphonemes → syllable structureAlternative-view auxiliaryorthography → phoneme classes
That makes it clear that you're not claiming all tasks contribute independent information. Rather, you're supplying multiple supervised views of the linguistic structures that classical G2P systems would either compute explicitly or rely upon implicitly.
From a speech-review perspective, that's a much more persuasive argument than trying to claim every auxiliary task is an independent source of signal. The goal is not additional information so much as additional structure. That's a distinction reviewers often appreciate once it's made explicit.

---

## User

My syllabication is on the phonetic level

---

## Assistant

That makes a lot of sense, and it changes the argument somewhat.
If the task is phonetic syllabification rather than orthographic syllabification, then it's not really an intermediate step toward generating phonemes. In a classical pipeline, you generally need the pronunciation before you can reliably syllabify at the phonetic level.
So the relationship is more like:
orthography     ↓phonemes   ↙   ↘stress  syllabification
rather than:
orthography     ↓syllabification     ↓phonemes
That means phonetic syllabification is probably best viewed as:


a derived representation of the target pronunciation,


an alternative view of phonological structure, or


a multi-view supervision signal,


rather than a pipeline component that precedes G2P.
That's not a problem. In fact, it may be quite useful.

For example, suppose your target pronunciation is:
photographF OW1 T AH0 G R AE2 F
The syllabification task might be:
F OW1 . T AH0 . G R AE2 F
or
[F OW1] [T AH0] [G R AE2 F]
The model is being encouraged to learn something about:


permissible onsets,


codas,


syllable nuclei,


stress-bearing positions,


without having to infer those structures purely from the phoneme sequence loss.
In that sense, syllabification becomes analogous to the syntactic auxiliary tasks used in older NLP MTL work: it's exposing latent structure that is already present in the target.

Actually, if I were mapping your tasks to the "traditional G2P pipeline" story, I'd separate them into two groups:
Upstream / pipeline-inspired tasks
These are things that a classical system might compute before or during pronunciation generation:


morphological segmentation


affix identification


grapheme classification


stress prediction (arguably)


lexical category information


Phonological structure tasks
These are analyses of the generated pronunciation:


phonetic syllabification


phoneme classes


sonority profile


onset/coda decomposition


stress pattern extraction


These are not necessarily pipeline stages, but they encode linguistic structure that a rule-based system would contain implicitly in its phonological rules.

One thing that may actually appeal to a speech audience is to frame the latter as factorizations of the pronunciation space.
Instead of predicting only:
word → phoneme sequence
the model is also trained to predict:
word → syllable structureword → stress structureword → phonological class sequence
all of which are projections of the same underlying pronunciation.
That's very close to the intuition behind old phonological rule systems, where pronunciation wasn't treated as an undifferentiated string but as a structured object with syllables, stress, feature classes, and so on.
So I'd probably describe phonetic syllabification not as a pipeline component, but as a structured phonological view of the target pronunciation. That feels both linguistically accurate and easy to defend experimentally. If it helps, you can even formulate the paper's task inventory as "pipeline-derived tasks" versus "phonological-structure tasks"; reviewers will immediately understand why the latter can be generated automatically from the pronunciation lexicon.

