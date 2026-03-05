Comments: This paper presents a benchmark dataset aimed at evaluating multimodal referential expression resolution within a 3D virtual environment featuring dyadic interactions.

The reviews are mostly positive about the work, especially the novelty of the benchmark.

There are some issues that the reviews are pointed out - in terms of clarity of annotation categories (positive NPs etc), details related to inter-annotator agreement, types of errors/error rates, etc. I presume all these details are at hand would be incorporated in the final draft. Please also thoroughly address the concerns in the reviews relating to tracking details b/w the actor + interlocutor.

============================================================================
                            REVIEWER #1
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                               Relevance: 5
                  Knowledge of the Field: 4
                               Soundness: 5
                                 Clarity: 3
             Originality of the Approach: 4
                 Significance of Results: 3
                           Replicability: No
                      Overall Assessment: 4
                     Reviewer Confidence: 3

Detailed Comments
---------------------------------------------------------------------------
Briefly describe what the submission is about:

The paper presents a benchmark for multimodal referential expression detection and resolution in a 3D environment. The dataset was collected from participant dyads following one of three scenarios in a virtual apartment through VR, and includes speech, motion, gaze, facial expressions, and 3D scene geometry. Compared to previous work, this dataset adds continuous dialogue with egocentric videos and full-body tracking. The rich annotations are created through a mix of automatic systems (LLMs, VLMs...) and manual correction work. A total of 6h of conversations is annotated with about 4k references. Authors also produce evaluations for two off-the-shelf systems as well as a modular pipeline they propose (GPT-4o+Florence-2).

Contributions:
- a dataset for multimodal grounding with rich features
- baseline evaluation results including an original combination

Strengths:
- Very clear objectives and work description
- Enabling dataset for research on embodied interaction
- Original method for combining referential expression detection in text and its projection to the visual modality

Weaknesses:
- Documentation of certain aspects is not clear due to absence of appendices
- All parts of the dataset are not evaluated (what is the use of tracking or 3D scene?)
- It's not clear why the dataset needs to be evaluated with crowdsourcing, and whether those annotations could be used in future research

Additional comments:
- Supplemental material, which includes detailed description of the collected data, is not available at time of review
- Footnotes 1 and 2 on page 4 are identical (and footnote 2 is used in a context where we expect something different)
- It would be interesting to see prompts for annotations with LLMs
- An example of annotations on a sample scene would help understand the scope of annotations
- It is not clear whether referential expressions detected by GPT4o are manually corrected or not
- What as annotator agreement on the different annotation levels? What is the estimated accuracy of automatic annotation stages on the target data?
---------------------------------------------------------------------------



---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                          Ethical Issues: No[4:55 PM]============================================================================
                            REVIEWER #2
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                               Relevance: 5
                  Knowledge of the Field: 4
                               Soundness: 4
                                 Clarity: 2
             Originality of the Approach: 3
                 Significance of Results: 4
                           Replicability: No
                      Overall Assessment: 3
                     Reviewer Confidence: 5

Detailed Comments
---------------------------------------------------------------------------
The paper describes a dataset for referential understanding in multi-modal situated communication between two interlocutors, an instruction giver (the main actor) and a follower (the interlocutor). The interactions take place in VR, in simulated apartment environments.
Reference resolution is a crucial aspect of natural language understanding, and in situated dialogue non-verbal, multi-modal cues are key to reference resolution. Therefore, providing a dataset of situated face-to-face communication is a valuable resource for researching multi-modal reference resolution. The employed VR setup in an instructor-follower scenario is a good means to collect realistic data on referring expressions. The categorization of referring expressions in NPs that uniquely describe a referent, NPs that partially describe a referent and pronouns makes sense. The technical set-up for the data collection is well described, and the availability of the dataset under cc by-nc 4.0 appropriate for a research data set.
With 4K referring expression-reference links and related multimodal contexts, the dataset is sufficiently large to perform various kinds of analyses and experiments based on it.
Despite the relevance of the dataset for research on language grounding, the presentation of the work leaves a number of questions unanswered that need to be clarified:
As regards the classification of referring expressions, the term "partitive NP" is misleading as the respective referring expressions are underspecified indicators of a referent rather than partitive nouns in a linguistic sense. Overall, a more precise definition of the categorisation of referring expressions in Sec. 3.2.2 would be helpful. For instance, as late as in Sec. 4.1 the reader learns that the "partitive NP" category also includes local referring expressions such as "there".  Thus, one can guess that temporal referring expressions are also subsumed by this category. The authors should explain why they refrain from introducing a fourth category representing temporal and spatial referring expressions.
Speech transcription and reference annotation are performed semi-automatically employing WhisperX and GPT-4o for topic annotation and the classification of referring expressions. Do I understand correctly that topic identification means that each referring expression in a transcribed segment of speech is mapped to an object present in the visual scene? If yes, this should be stated very clearly in the paper.
It is also not clear to me why topic annotation and classification of referring expressions is modeled as a pipeline. To me it seems that the processes are independent of each other and can be done in parallel. Is this correct? If yes, talking about a multi-stage annotation pipeline in Sec. 3.2 is misleading.
What is the error rate of the LLM regarding the two annotation tasks? It is not clear who did the manual verification of the links between the referring expressions and visual referents? Was this also done via crowd sourcing as described in Sec. 4.2, or was this done by researchers involved in developing the dataset? How many annotators did manually verify the links? What was the error rate of the humans? Was one link verified by more than one person? If so, what was the intercoder agreement?
An example of a fully annotated multi-modal utterance would be helpful too.
As it seems, the main actor is more extensively captured than the interlocutor, and it is not fully clear which modalities are captured for the interlocutor. As of Table 9, it seems that only the interlocutor's motion is captured. Please, clarify and explain why the interlocutor is less captured than the main actor. As of Sec. 3.3, it seems that the interlocutor's perspective is not annotated at all. Why is this so? In situated dialogue reference resolution is relevant for the interlocutor to understand the message of the main actor, therefore the perspective of the interlocutor (the addressee) is particularly crucial. Explain in this context why the focus of the data lies on the 1st person perspective of the main actor.
As regards the experiments in Sec. 4, it seems that the human evaluation task in Sec. 4.2 is designed such that it provides a comparison to the VLM experiment in Sec. 4.3. If yes, this should be made more explicit and discussed, because otherwise one is wondering why the human task is main actor centric and not interlocutor centric, because the interlocutor uses the multi-modal input to interpret the main actors message.

Further comments/questions:
Synchronization of data streams: Briefly explain why you use classical SMPTE and not another time code.
In Tables 3 and 4, it is unclear what the group numbers stand for.
In Tables 5 and 6, explain what single and multiple stand for, as well as 3-7 in the pid column.
Moreover, is there a relation between the numbers in the pid and the group columns of Tables 3, 5 and 6?
On page 6, it is stated that "Main speakers more often introduced objects into the conversation, yielding relatively more full mentions, whereas the interlocutor tended to react to already-salient items, relying heavily on pronominal and deictic forms". Could it be that this is also an artifact of the instruction to the interlocutor to not introduce new referents?
---------------------------------------------------------------------------



---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                          Ethical Issues: No[4:55 PM]============================================================================
                            REVIEWER #3
============================================================================

---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                               Relevance: 4
                  Knowledge of the Field: 4
                               Soundness: 4
                                 Clarity: 5
             Originality of the Approach: 4
                 Significance of Results: 4
                           Replicability: Yes
                      Overall Assessment: 4
                     Reviewer Confidence: 2

Detailed Comments
---------------------------------------------------------------------------
Summary:
The manuscript introduces a novel benchmark "Speak, Point, Look" to assess how well VLMs can actually identify referential expressions in conversations in 3D environments. The authors collected data from 6.7 hours of interaction between a speaker and a listener while role-playing in a VR environment. Among these, 4,211 referential expressions in the utterances were selected and annotated into three types: full NP, partitive NP, and pronominal. Afterwards, mappings with actual 3D objects were manually verified, resulting in a final set of 4,001 references as a benchmark.

Evaluations were conducted on both humans and VLMs. "Humans performed best on explicit noun phrases without context (though adding context slightly reduced performance for them), but context significantly improved resolution of ambiguous partitives and pronouns Meanwhile, the authors compare end-to-end models like Florence-2 and GroundingGPT with the authors' newly proposed modular pipeline. While single models perform well on explicit noun phrases, they exhibit limitations in properly utilizing conversational context when representing partitives and pronouns. In contrast, the modular pipeline outperforms the baseline, particularly on pronouns.
Pros:
1.	Well-made 3D conversational grounding benchmark: The paper introduces a novel benchmark from unscripted VR role-play, where referential expressions emerge naturally in dialogue. This would be very important resource for robotics, simulation and game environment.
2.	Rich multimodal cues and large size of finegrained annotated data: The dataset tightly aligns speech, egocentric RGB/depth, and 3D object instances, and further categorizes 4,000+ referring expressions into full NPs, partitives, and pronominals with manual verification.
3.	Clear research question and strong baselines: By comparing human performance, end-to-end VLMs, and a modular pipeline, the paper clearly shows that current VLMs struggle with pronominal references and that a simple two-stage modular approach can handle this issue.

Cons:
1.	Limited domain: The data is collected in a small number of VR apartment scenes, which only cover limited scenarios.
2.	Snapshot-based evaluation: Adding an evaluation setting that explicitly tackles fully temporal grounding would further strengthen the impact of this work.

Other comments:
1.	Due to the  line breaks, Table 1 and 2 are hard to read. The aurthors may want to enhance presentation of those two tables.
2.	How many kinds of pronominal references are contained in this work?