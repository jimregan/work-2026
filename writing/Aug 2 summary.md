
# Chat Conversation

*Exported on 19/08/2026, 15:06:38*

---

## User

I'd like a summary of this paper; it should be sufficient so I can later compare it to the implementation

---

## Assistant

The paper’s central claim is stronger than merely “S3M embeddings contain phonetic information.” It argues that phonological contrasts correspond to approximately linear directions in representation space, that these directions are compositional, and that moving different distances along them produces graded, acoustically interpretable changes. The headline example is:
r[b]​≃r[p]​+(r[d]​−r[t]​)
so r[d]​−r[t]​ behaves as a voicing direction which can be applied to a bilabial stop. The second experiment turns this descriptive claim into an intervention: add λv to selected WavLM frames, invert the modified representation with a vocoder, and see whether acoustic properties vary systematically with λ. 2602.18899v3
1. Data and representations
They use TIMIT for English and VoxAngeles for cross-linguistic evaluation. VoxAngeles contains 95 non-English languages from 21 families, so the paper describes the overall study as covering 96 languages. TIMIT diphthongs are excluded, and its separately labelled stop closures/releases are merged to make segments closer to IPA phone units. VoxAngeles is already manually phonetically segmented. 2602.18899v3
For the main representation experiments they compare three English-trained LARGE S3Ms: wav2vec 2.0, HuBERT, and WavLM. Each is treated as 7 convolutional feature-encoder layers followed by 24 Transformer blocks, roughly 300M parameters. They extract 25 representations: layer 0 is the CNN output and layers 1–24 are the Transformer outputs. The effective S3M stride is 320 samples at 16 kHz, i.e. about 20 ms/frame. Spectral baselines are librosa-default MFCCs and log-mel spectrograms. 2602.18899v3
The main phone representation is feature-sliced, not audio-sliced. They first run the complete utterance through the model,
R=f(x),
convert a phone interval (ts​,te​) to representation indices
ts′​=⌊ts​/s⌋,te′​=⌈te​/s⌉,
then mean-pool
r=avgpool(R[ts′​:te′​]).
That distinction matters: the vector for a phone is therefore computed from hidden states whose Transformer receptive field includes the surrounding utterance. An appendix compares this with cropping the waveform first and finds that feature slicing works much better for S3Ms. The authors explicitly interpret this as evidence that contextual information contributes to the phonological geometry. 2602.18899v3
2. Experiment 1: testing vector arithmetic
Each phone is represented phonologically with PanPhon’s 21 ternary features, originally {−1,0,+1}. For construction of the analogies, each ternary feature is expanded to two binary dimensions:
+→[1,0],0→[0,0],−→[0,1],
giving a 42-dimensional vector hp​.
They then exhaustively search the available phone inventory for quadruplets
(p1​,p2​,p3​,p4​)
satisfying
hp1​​−hp2​​=hp3​​−hp4​​.
Thus the phonological difference between p1​,p2​ must exactly equal that between p3​,p4​. Importantly, these do not have to be minimal pairs: a relation may involve several feature changes simultaneously. The representation-space prediction is
rp1​​≃rp2​​+rp3​​−rp4​​.
Phones without PanPhon mappings or with fewer than 50 occurrences are discarded. This reduces TIMIT from 47 to 43 usable phones and VoxAngeles from 567 to only 57. They obtain 236 TIMIT quadruplets and 468 VoxAngeles quadruplets. “Consonantal” and “constricted glottis” yield no usable quadruplets in either corpus, leaving 19 PanPhon features represented in the analogy tests. 2602.18899v3
The actual success criterion
This is important when comparing with code: they do not principally evaluate whether arithmetic produces the nearest embedding of the target phone.
For a quadruplet they estimate
cos(p)=E[cos(rp1​​,rp2​​+rp3​​−rp4​​)].
Phone tokens are repeatedly sampled rather than replacing each phone with one global centroid. They compare this to:
cos+(p)=E[cos(rp1​​,rp1​′​)]
for two instances of the same phone, and
cos−(p)=E[cos(rp1​​,rother phone​)].
The desired ordering is
cos−<cosanalogy​<cos+.
They bootstrap by sampling 1000 phone instances with replacement, calculate an averaged cosine, and use 10 bootstrap replicates to obtain 99% confidence intervals. Operationally, they require separation of the CIs, not merely the point-estimate ordering: the upper CI of the lower quantity must lie below the lower CI of the higher one. “Success rate” is the proportion of quadruplets satisfying that criterion. 2602.18899v3
So if the implementation simply computes phone centroids, performs b−a+c, and asks whether the correct phone is nearest, it is implementing a materially different experiment.
3. Experiment 1 results
On TIMIT, peak success rates are approximately:
RepresentationBest success rateHuBERT94%, final layerWavLM92%, final layerwav2vec 2.061%, middle layerMFCC19%log-mel0%
On VoxAngeles, WavLM is much stronger than the other S3Ms: WavLM 93%, HuBERT 45%, wav2vec 2.0 39%, MFCC 19%, log-mel 0%. Of the 468 VoxAngeles analogies, 316 (68%) contain at least one phone absent from TIMIT, despite all three principal S3Ms having been trained on English. 2602.18899v3
WavLM shows three broad regions of success across its depth. Vowels tend to give an earlier intermediate peak; consonants peak later; both become very strong at the final layer. The authors interpret this in terms of temporal context: many vowel cues are relatively local whereas consonantal properties can be distributed through release, neighbouring formant transitions, aspiration, devoicing, etc. Audio slicing, which removes most of that context, substantially weakens the S3M effect. 2602.18899v3
There is an important robustness qualification in the appendix. Their secondary offset-based analysis uses Fournier et al.’s pairing consistency score rather than the item arithmetic above. Phone pairs are grouped by identical PanPhon differences; genuine relation offsets are contrasted against offsets produced by shuffling the second phone, and ROC AUC is reported. This broadly confirms that S3Ms outperform spectral features, but its layerwise behaviour differs: intermediate layers often look best rather than WavLM’s enormous final-layer peak. They make the item test primary partly because the offset test leaves only 36 TIMIT and 112 VoxAngeles quadruplets. 2602.18899v3
There are several further controls worth remembering. Audio-sliced MFCCs surprisingly reach roughly 67% on TIMIT and 50% on VoxAngeles, but their cosine space is extremely anisotropic, with similarities collapsing near 1; WavLM avoids this collapse. Fine-tuning XLSR-53 for phone recognition strengthens the analogy structure. Neither individual PanPhon feature nor the number of feature differences between phones explains the main layerwise pattern. 2602.18899v3
4. Experiment 2: constructing explicit phonological steering vectors
The second experiment uses final-layer WavLM, not arbitrary analogy offsets.
For each feature i, they estimate one global vector:
vi​=Ehi​=+1​[r]−Ehi​=−1​[r].
This is a crucial distinction from the paper title’s [d]−[t]: the actual synthesis experiments normally use difference-of-class-means vectors over many phone tokens, rather than a single minimal pair.
They calculate vectors separately for vowels and consonants: vowel vectors come from vowel representations and consonantal vectors from consonants. Their later sample-efficiency experiment suggests a few hundred samples per side are sufficient: by N=256, a subsampled vector is generally very close to the full-data vector. Conversely, extracting a vector from a single phone pair tends to have only about 0.5 cosine similarity to the full vector, despite potentially hundreds of tokens of each phone, because that pair also carries other correlated phonetic differences. 2602.18899v3
They modify only frames belonging to the selected phone:
R~t​={Rt​+λvi​,Rt​,​ts′​≤t<te′​otherwise.​
The surrounding hidden states are left untouched.
For TIMIT, the feature vector is estimated from the training split and evaluation uses the test split. For VoxAngeles, which has no standard split, they use a fixed randomly selected subset of languages to estimate the vector and evaluate on the remaining languages. The paper does not identify those languages in the prose, so that is one thing that will have to come from the implementation. 2602.18899v3
For each of the eight tested vectors they resynthesize 3000 examples, randomly selecting a target segment with replacement and sampling
λ∼U(−5,5).
Thus the experiment intentionally tests substantial extrapolation as well as the [−1,1] region one might regard as interpolation.
5. WavLM inversion
They train a Vocos-based vocoder to approximate f−1, i.e. waveform reconstruction directly from the WavLM representation.
There are two versions: an English vocoder trained on LibriTTS, 585 h, and a multilingual vocoder trained on FLEURS-R, about 1.3k h across 102 languages. They alter Vocos’s architecture dimensions to accept WavLM (and, in an ablation, MFCC) inputs, and state that they multiply both the original Vocos batch size and learning rate by eight for faster convergence. The paper does not state the resulting absolute LR, batch size, precise architecture dimensions, training duration, optimizer configuration, etc.; those will need to be recovered from the implementation rather than treated as specified by the paper. 2602.18899v3
They also run an identity reconstruction control,
x~=f−1(f(x)),
with λ=0, and show that the resulting changes in the acoustic measures are tightly centred around zero. This is meant to argue that the correlations below are not simply vocoder reconstruction artefacts.
6. Acoustic tests of vector scale
Eight features have acoustic proxies:
PanPhon directionMeasurementClaimed expected λ-correlationhighF1−lowF1+backF2−roundF2−nasalF1 bandwidth−sonorantHNR+stridentspectral centre of gravity+voicespectral centre of gravity−
Measurements are implemented through Parselmouth/Praat. F1 represents height; F2 backness and rounding; F1 bandwidth nasality; HNR sonority; and spectral centre of gravity voicing and stridency. The analysis correlates λ with the change in the relevant acoustic measurement using Spearman’s ρ. 2602.18899v3
There is one paper-internal detail I would specifically keep in mind when we inspect the code later. Appendix A.3 says nasalization produces broader F1 bandwidth, which naturally sounds like a positive relation with increasing +nasal vector, but Table 1 explicitly predicts a negative correlation and Figure 4 indeed reports negative correlations. The prose, table and vector definition are therefore not transparently consistent on the sign of the nasality measurement. I would check the implementation of both the F1BW measure and the definition of Δ before interpreting it.
On TIMIT the reported Spearman correlations are strong and have the expected signs according to their Table 1. They report the effect separately for phones already possessing vs lacking the feature; representative pairs of ρ are high −.801/−.826, low +.908/+.836, back −.759/−.733, round −.833/−.853, nasal −.441/−.472, sonorant +.649/+.659, strident +.819/+.788, and voice −.720/−.794. 2602.18899v3
The relationship need not itself be linear: the claim is that a linear change in representation space produces a monotonic change in the acoustic variable.
7. What the interventions actually sound/look like
The qualitative examples on pp. 7–8 are useful because they make clear that the vectors are not claimed merely to change a static spectrum.
Adding the rounding vector to English [i] lowers its formants, including F2/F3, creating cues toward a front rounded vowel even though English lacks that category. Increasing the voicing vector on [b] moves voicing onset progressively earlier, eventually extending voicing into the closure and producing negative VOT. Increasing stridency on [b] introduces high-frequency frication and simultaneously removes the stop burst. Increasing nasality weakens/removes the burst and adds a low-frequency nasal murmur. These are temporally organised alterations rather than just spectral offsets. 2602.18899v3
The effects remain interpretable even for ∣λ∣>1, although some directions saturate. The paper mentions already-sonorant segments, voiced consonants, and non-stridents as cases where extrapolation eventually ceases to give much additional change.
The multilingual experiment repeats the procedure with held-out VoxAngeles languages and the FLEURS-R vocoder and gets very similar monotonic trends. A rounding case study also finds that the steering vector mirrors dataset-specific acoustic manifestations: in TIMIT rounding differs most conspicuously in F2/F3, whereas in VoxAngeles F1/F2 carry more of the contrast, and the intervention reproduces that pattern. 2602.18899v3
8. What I would regard as the implementation contract
When we compare the repository with the paper, the high-value things to check are: PanPhon ternary-to-42D conversion and exact quadruplet enumeration; the ≥50-token filter and phone normalisation; full-utterance feature slicing with floor/ceil boundary conversion; token-level random sampling rather than only phone centroids in Experiment 1; 1000 samples × 10 replicates and the 99% CI separation rule; whether “236/468 quadruplets” means the same permutation/deduplication logic as in the code; all 25 model layers; the alternative PCS implementation; final-layer WavLM for synthesis; +feature mean minus −feature mean with vowel/consonant separation; the precise TIMIT and VoxAngeles vector/evaluation splits; 3000 samples per vector with λ∼U(−5,5); modification of target frames only; Vocos input/training changes; and, especially, the exact acoustic-measurement and delta/sign conventions.
The main conceptual caveat is also useful for reading the code: Experiment 1 and Experiment 2 do not construct “the phonological vector” in exactly the same way. Experiment 1 establishes arithmetic using arbitrary matched phone quadruplets; Experiment 2 deliberately switches to a much more stable difference-of-class-means vector. The appendix shows that a single phone-pair vector is in fact a fairly poor estimate of that global vector.
Finally, the authors themselves limit the conclusion: they test only a few S3Ms, only PanPhon’s feature system, and only one vocoder family. They explicitly say the experiments cannot distinguish “the model has discovered a phonological natural-class feature” from “the model has captured a sufficiently consistent acoustic/phonetic difference.” The resynthesis results are also jointly properties of WavLM and the trained inverse model. 2602.18899v3
That should give us a fairly concrete specification against which to read the repository: not just what the paper claims, but what data flow and numerical operations the implementation ought to contain.

---

## Assistant

This is a good implementation-oriented summary, and it lines up closely enough with the paper that it should be useful as the second side of the comparison. A few details are worth correcting or sharpening before treating it as authoritative.
The clearest terminology issue is pcs.py: PCS is “pairing consistency score,” not “Phonological Consistency Score.” The paper takes this from Fournier et al. and defines it as separability between offsets from the same relation and mismatched offsets, evaluated by ROC AUC; random performance is 0.5. 2602.18899v3 The union-find detail is implementation-specific and perfectly useful in a code summary, but it is not part of the method description in the paper.
The analyze_synth.py description is also slightly broader than the published experiment. The paper’s eight principal acoustic tests use only F1, F2, F1 bandwidth, HNR, and spectral centre of gravity: height→F1, back/round→F2, nasal→F1BW, sonorant→HNR, voice/strident→COG. 2602.18899v3 If the script also computes zero-crossing rate and RMS, those are extra implementation facilities rather than measurements supporting the main claim. I would phrase that distinction explicitly. MFCC synthesis, however, genuinely is part of the paper: Appendix B.13 repeats the intervention experiment with feature- and audio-sliced MFCCs and finds that their vectors are mostly ineffective. 2602.18899v3
The workflow diagram slightly obscures an important representation distinction. extract_features.py may produce “phone-level representations,” but the synthesis experiment requires the unpooled frame sequence R, because the intervention is applied only to frames within the phone interval before decoding. By contrast, the analogy experiment consumes pooled phone vectors r. So conceptually I would represent that fork as something like:
extract_features.py   |   +--> pooled phone vectors --------> estimate_similarity.py / pcs.py   |   `--> framewise representations ---> analyze_synth.py
That distinction matters when checking whether the implementation really corresponds to equations (10) versus (16) in the paper.
There is also a potentially important point hidden by “uses PanPhon feature vectors to discover valid analogies.” The main analogy code ought to reproduce the paper’s rather specific construction: PanPhon’s 21 ternary features are expanded to 42 binary dimensions, and quadruplets are accepted when
ha​−hb​=hc​−hd​.
That exact encoding is what makes 0 distinct from both + and −. 2602.18899v3 If the code instead compares ordinary 21-dimensional −1,0,+1 difference vectors, that would not be strictly equivalent, so this is one of the first things I would inspect in estimate_similarity.py.
Likewise, “repeated Monte Carlo sampling and 99% confidence intervals” is correct but underspecified relative to the paper. The published procedure draws 1000 phone representations with replacement per estimate, uses 10 replicates, and tests the analogy by CI ordering:
cos−<cosanalogy​<cos+.
That is a sufficiently peculiar procedure that the implementation should be checked line-for-line against it rather than merely described as bootstrapping.
A few other codebase points correspond especially closely to appendix experiments rather than the main body. Both feature-vs-audio slicing and MFCC synthesis are explicit appendix studies; feature slicing is the primary S3M setup, whereas audio slicing removes neighbouring context and is mainly a control. The paper actually finds audio slicing helps MFCC substantially while hurting S3Ms. 2602.18899v3 Similarly, pcs.py corresponds to Appendix B.1 rather than the principal success-rate metric.
One thing your summary already makes much clearer than the paper is where implementation assumptions live. The paper says VoxAngeles has 95 languages and that a random subset of languages is used for synthesis-vector estimation versus evaluation, but it does not expose the exact language-selection machinery. 2602.18899v3 If the repository literally assumes exactly 95 languages, or reconstructs the split by integer indices/random seed, that is an implementation fact we will want to distinguish from the scientific method.
So I would retain this summary, with three edits:


rename Phonological Consistency Score → pairing consistency score;


distinguish the paper's five reported acoustic measurement types from extra diagnostics such as ZCR/RMS;


distinguish pooled phone embeddings used by the analogy tests from framewise representations needed for steering/resynthesis.


The interesting next comparison is now quite concrete. I would read estimate_similarity.py first, because that is where there is the greatest scope for the prose description “implements the paper” to conceal consequential details: PanPhon encoding, quadruplet generation/deduplication, sampling, random-phone baseline, CI calculation, and success criterion. After that, analyze_synth.py is the other high-value file, particularly for the train/test language split, definition of Δ, and the suspicious nasality/F1BW sign issue from the paper.

---

## User

Next one; I'd like a (standalone) summary of this paper

---

## Assistant

Self-Supervised Speech Models Encode Phonetic Context via Position-dependent Orthogonal Subspaces
Kwanghee Choi, Eunjung Yeo, Cheol Jun Cho, David R. Mortensen, and David Harwath, 2026. 2603.12642v1
Central idea
The paper asks what it actually means for Transformer-based self-supervised speech models (S3Ms) to produce contextualized frame representations.
Its proposed answer is quite specific. A representation at one time step does not merely encode the phone acoustically aligned with that frame. It contains phonological information about a sequence of phones around that frame, but keeps information about different relative positions separate.
Schematically, a frame within phone p0 is proposed to contain something like
rt​≈vp−1−1​+vp00​+vp+1+1​+⋯
where the superscript indicates relative phone position. Crucially,
vvoice−1​
and
vvoice0​
are not simply the same voicing vector added twice. The authors argue that phonological information for different relative positions occupies approximately orthogonal subspaces. Thus the representation can distinguish “the previous phone is voiced” from “the current phone is voiced.”
The second major claim is that those relative positions are defined with respect to phonetic segments, rather than fixed amounts of elapsed time. As the signal crosses a phone boundary, the same underlying phone changes from “current” to “previous,” etc., and the corresponding representation shifts between positional subspaces. The resulting changes line up with manually annotated phone boundaries. 2603.12642v1
So the intended representational picture is:
one frame=phonology of current phone+phonology of neighbouring phones,
with each phone-position combination represented in its own approximately orthogonal coordinate system.

Models, data, and baseline representations
They analyse the LARGE versions of three English-trained S3Ms:


wav2vec 2.0 (facebook/wav2vec2-large-lv60)


HuBERT (facebook/hubert-large-ll60k)


WavLM (microsoft/wavlm-large)


All have a convolutional feature encoder followed by a Transformer context network. MFCC and log-mel representations serve as non-contextual spectral baselines.
The datasets are TIMIT, manually segmented English speech, and VoxAngeles, manually segmented speech from 95 non-English languages. The latter is particularly useful because the tested S3Ms were trained on English, so it tests whether the proposed structure extends to phonetic inventories outside the models' training language. 2603.12642v1
Their principal diagnostic is phonological vector arithmetic, with analogies automatically generated from PanPhon features. For example, if the relevant representation geometry is linear, a relationship such as
[b]−[d]+[t]≈[p]
should hold.
An analogy counts as successful when the cosine similarity between the target and arithmetic approximation falls between a different-phone lower baseline and a same-phone upper baseline. They estimate similarities using 1,000 samples with replacement and construct 99% confidence intervals from 10 independent repetitions.
Phones occurring fewer than 50 times are excluded for these analogy experiments, leaving 43 TIMIT phones and 57 VoxAngeles phones, producing respectively 236 and 468 valid phonological quadruplets. 2603.12642v1

1. Is phonological structure present in a single frame?
The first experiment establishes that the earlier phonological-vector results are not an artefact of averaging representations across an entire phone.
They compare:


mean pooling: average all S3M frames corresponding to the phone;


center pooling: take only the single frame at the temporal centre of the phone.


Surprisingly, center-frame representations perform at least as well as mean-pooled representations and frequently perform better. The effect appears for both TIMIT and VoxAngeles and across the S3Ms, although the layerwise behaviour varies considerably by model.
This supports the claim that phonological compositionality genuinely exists at the individual Transformer frame level: it is not necessary to average several frames to reveal something resembling a stable phone representation.
Spectral MFCC/log-mel representations do not show the same strong behaviour. 2603.12642v1

2. Does that single frame encode neighbouring phones?
The key experiment then changes what the analogy is about.
Suppose the local phone sequence is
[p−2,p−1,p0,p+1,p+2]
and the representation examined is a single center frame belonging to p0.
Instead of asking only whether that frame supports an analogy concerning p0, they ask whether its representation supports analogies concerning each of the five relative positions.
In other words, can a frame sitting inside one phone encode distinctions such as:


“the previous phone is voiced”;


“the current phone is voiced”;


“the next phone is voiced”?


It can, particularly in later S3M layers.
The strongest analogy success occurs for p0, as expected, but useful phonological information also appears for p−1 and p+1. Positions p−2 and p+2 generally show little success by this metric.
This is therefore not just a conventional receptive-field result showing that the Transformer has access to neighbouring audio. The claim is that the information about neighbouring segments occurs in a representation sufficiently regular to support the same kind of phonological vector arithmetic used for the current segment. 2603.12642v1
Effective context
They examine this more finely by randomly selecting individual frames at normalized positions through each phone from p−2 to p+2.
WavLM gives the clearest result: a broad trapezoidal context profile. Information about the target phone is strong inside that phone, weaker in the immediately adjacent phones, and approaches zero farther away. HuBERT behaves similarly on TIMIT but less clearly on VoxAngeles; wav2vec 2.0 shows substantially weaker contextual effects.
MFCC and mel features behave as expected for acoustic-local representations: their useful information is essentially restricted to the target phone.
Thus contextualization is strongest in WavLM, then less consistently apparent in HuBERT and wav2vec 2.0. 2603.12642v1

3. Position-dependent phonological subspaces
The central problem is now obvious.
Suppose a frame encodes:


current phone = voiced;


previous phone = voiced.


If both are represented by simply adding the same vector vvoice​, the representation cannot say which phone is voiced.
The authors therefore hypothesize separate positional versions:
vvoice−1​,vvoice0​,vvoice+1​.
They construct such vectors through differences of class means. For example,
vvoice0​=E[r∣p0 voiced]−E[r∣p0 unvoiced]
where r is still the center-frame representation of p0.
For the previous-phone version,
vvoice−1​=E[r∣p−1 voiced]−E[r∣p−1 unvoiced].
Notice something important: both vectors are extracted from representations at the same temporal location. The grouping criterion changes according to which surrounding phone possesses the feature.
They do this for eight features:


vowel: high, low, back, round;


consonant: nasal, sonorant, strident, voice.


The main analysis uses final-layer WavLM. 2603.12642v1
Within each positional space
The expected phonological geometry is preserved. For instance:


high and low vectors are strongly negatively correlated;


nasal, sonorant and voice are positively related;


strident shows negative relationships with several sonorant-type features;


vowel and consonant vectors are approximately orthogonal.


This structure is similar in TIMIT and VoxAngeles.
More importantly, a broadly similar pattern appears independently for p−2,p−1,p0,p+1,p+2.
Between positional spaces
Vectors belonging to different relative positions have much smaller cosine similarities.
Thus
cos(vvoice0​,vvoice−1​)
is much lower than similarities among phonological vectors derived for the same relative position.
The paper calls this positional orthogonality: roughly the same phonological system appears to be replicated in largely separate representational directions for each neighbouring phone position. 2603.12642v1
The effect also appears across layers and across all three models when comparing average absolute cosine similarity for same-position versus different-position vector pairs.
One qualification is important: the paper does not estimate complete linear subspaces and measure their principal angles. “Orthogonal subspaces” is inferred from the relatively low cosine similarities between sets of difference-of-means phonological vectors. The empirical result is therefore more directly “position-indexed phonological directions are approximately orthogonal” than a proof that the full representational spaces are mathematically orthogonal.

Information appears to decay with distance
There is an interesting discrepancy between the two diagnostics.
The direct analogy test finds little evidence at ±2 phones. Yet the difference-of-means method still recovers recognizable position-dependent phonological structure at ±2.
The authors argue that the analogy success metric is therefore relatively insensitive: contextual information from several sources is superposed within a frame and individual-vector arithmetic becomes noisy.
Supporting this interpretation, the L2 norms of position-dependent phonological vectors decline approximately as
0>∣±1∣>∣±2∣.
So representations apparently contain weaker traces of progressively more distant phones rather than abruptly cutting off after the immediate neighbours.

4. The positional representation follows phone boundaries
The paper's next claim is stronger than “the Transformer has an effective temporal window.”
If p−1,p0,p+1 really refer to phones, then the model must in some sense track where phones begin and end. Otherwise the positional encoding could merely correspond to something like ±100 or ±200 ms.
They therefore examine 11 consecutive S3M frames around manually annotated TIMIT phone boundaries: five before the boundary, the boundary frame, and five afterwards.
They select boundaries where a particular phonological feature changes. For instance,
[−voice]→[+voice].
Before the boundary, a vector meaning approximately “the next phone is voiced” should provide evidence for the upcoming voiced phone. After crossing the boundary, that same segment is now current, so “the current phone is voiced” should dominate.
That is approximately what happens.
Across the eight tested features, the cosine curves representing neighbouring and current positional vectors cross around the manually annotated TIMIT boundary. The reverse transition occurs at segment offsets: information formerly encoded as current becomes information about the preceding phone. 2603.12642v1
The visualisation on page 7 makes the intended result especially clear. Across an entire TIMIT sentence, cosine similarities to the −2,−1,0,+1,+2 feature vectors form a staircase-like pattern, with abrupt transitions repeatedly lining up with phone boundaries. 2603.12642v1
The authors therefore argue that the relevant contextual representation is more like:

previous segment / current segment / following segment

than:

100 ms ago / now / 100 ms later.

A necessary qualification is that this is not itself an unsupervised segmentation evaluation. Ground-truth TIMIT boundaries are used both to define the positions and to aggregate the analysis. What they show is that a representational signal changes in systematic alignment with those boundaries, which could potentially be exploited by a future unsupervised segmenter.

5. Connection to masked prediction
The paper also offers a possible reason why this structure emerges.
wav2vec-style S3Ms use context to predict masked input. Because speech is coarticulated, reconstructing a missing region benefits from knowing phonetic information on either side.
They test mask reconstruction by comparing representations produced from an intact signal with representations for the same region when it has been masked. The default mask length is 10 frames, approximately 205 ms. Since Transformer spaces are anisotropic, they first perform ZCA whitening before computing cosine similarities.
For HuBERT and WavLM, similarity between the true and context-reconstructed representations steadily improves through the network. wav2vec 2.0 instead reaches its best reconstruction similarity in an intermediate layer and then deteriorates.
This qualitatively resembles the models' different layerwise phonological-context behaviour. The authors therefore suggest—not demonstrate causally—that masked-prediction training may encourage frames to store structured information about neighbouring phones because that information aids reconstruction. 2603.12642v1

Interpretation
The paper proposes a fairly concrete picture of Transformer contextualization in speech:
rt​≈j=−k∑k​Vj​ϕ(pj)
where ϕ(pj) is something like the phonological feature content of the phone at relative position j, and Vj​ places that content into a position-specific representational subspace.
This accounts simultaneously for several observations:


A single S3M frame contains much more than its local acoustics.


It preserves phonological structure for the segment containing that frame.


It also contains weaker structured information about neighbouring segments.


The model can avoid confusing identical features belonging to different phones because relative positions are represented approximately orthogonally.


These positional assignments appear to change at phonetic boundaries rather than at fixed elapsed-time intervals.


The authors compare this informally with context-dependent triphones in traditional ASR: self-supervised models may effectively be rediscovering a richer continuous analogue of explicit left/current/right phonetic context. 2603.12642v1

Main implications and limits of the evidence
The result potentially gives a mechanistic explanation for several otherwise loosely connected properties of S3Ms: why learned discrete units resemble phonetic categories, why contextual S3M embeddings work well for phonetic tasks, why allophonic environment can be decoded, and why speech boundaries can emerge from otherwise unsupervised models.
The authors suggest applications to interpretable speech representations, unsupervised segmentation, structured discrete speech units, and architectures that explicitly disentangle context-independent from context-dependent information. They also speculate that phonetic position may be one example of a broader principle whereby S3Ms place different kinds of information—phonology, speaker identity, acoustics—into approximately separated linear subspaces. 2603.12642v1
The evidence nevertheless supports a narrower claim than the title might initially suggest. The strongest analyses concentrate on final-layer WavLM, particularly the orthogonality and boundary experiments; contextual analogy results are considerably weaker for wav2vec 2.0 and partly dataset-dependent for HuBERT. The “subspaces” are diagnosed through a small set of eight PanPhon-derived difference vectors rather than by characterizing all dimensions of a true subspace. And the apparent segmentation is evaluated relative to existing manual phone segmentation rather than obtained independently.
Within those qualifications, the paper's main result is quite striking: individual S3M frames appear to contain a structured representation of a short phonetic sequence, indexed by relative segment position, rather than merely a context-smoothed representation of the locally aligned phone. 2603.12642v1

---

## User

Next one. I realise that these papers are all connected, but I'd still like this to be a separate, self-contained summary

---

## Assistant

PRiSM: Benchmarking Phone Realization in Speech Models
Shikhar Bharadwaj, Chin-Jou Li, Yoonjae Kim, Kwanghee Choi, Eunjung Yeo, et al., 2026. 2601.14046v2
Central idea
The paper introduces PRiSM — Phone Realization in Speech Models, a benchmark for evaluating language-agnostic phone recognition (PR) systems.
Its starting argument is that conventional phone-recognition evaluation is too narrow. A model is normally judged by comparing its predicted phone sequence with a reference transcription, but this does not answer two distinct questions:


How accurately does the system represent the phone realizations actually present in the signal?


How useful is the phonetic information learned by the model for tasks that depend on pronunciation?


The distinction matters because a speech model exposes phonetic information through two quite different channels:
speech→{explicit phone transcriptionlatent speech representation​
A discrete IPA transcription is interpretable but inevitably discards information. Hidden representations retain far more acoustic detail, but may also contain speaker, semantic, prosodic, and other information unrelated to phonetics.
PRiSM therefore evaluates systems in two ways:


intrinsic evaluation: how good the generated phonetic transcription is;


extrinsic evaluation: how useful either the transcription or internal representation is for downstream speech tasks.


The paper is consequently as much an argument about how phone recognizers should be evaluated as it is a leaderboard comparison. 2601.14046v2

1. What counts as phone recognition?
The authors use phone recognition relatively literally: converting speech into units intended to describe physical phonetic realization, rather than recovering an underlying phonemic sequence.
For example, regional realizations of tell may receive different narrow phone transcriptions even though both correspond to the same phonemic form. This is why they regard PR as useful for cross-lingual phonetics, pronunciation analysis, clinical speech, and other situations where normalization to lexical or phonemic categories would remove the phenomena of interest. 2601.14046v2
They use “PR system” broadly enough to include any speech-to-IPA pipeline, ranging from dedicated CTC phone recognizers through encoder-decoder systems to general-purpose Large Audio Language Models (LALMs) prompted to produce IPA.
This breadth is intentional: one objective of PRiSM is to make very different system families comparable.

2. Benchmark structure
PRiSM consists of six intrinsic PR tests and nine downstream tasks.
Intrinsic: transcription quality
The six datasets are divided conceptually into two groups.
Variation within a seen language
These test whether a model can represent pronunciation variation rather than simply normalize the input toward familiar language-specific patterns:


PR-tmt: TIMIT — English;


PR-arc: L2-ARCTIC Perceived — non-native English;


PR-saa: Speech Accent Archive — accented/regional English.


Unseen languages
These test cross-lingual phonetic generalization:


PR-drc: DoReCo — 45 languages;


PR-vox: VoxAngeles — 95 languages;


PR-tsm: Tusom2021 — Tusom.


For closed LALMs, the authors correctly qualify “unseen”: their complete training data is unknown, so genuine absence from training cannot be verified. 2601.14046v2

3. Phonetic Feature Error Rate rather than PER
The principal intrinsic metric is Phonetic Feature Error Rate (PFER) rather than ordinary Phone Error Rate.
PER treats phones categorically:
[p]=[b]=[m],
so every substitution has essentially the same unit cost.
PFER instead maps IPA segments to articulatory/phonological features and computes edit distance in that feature representation. Informally, confusing phones that differ by one feature is therefore less serious than predicting something articulatorily quite different.
The corpus-level metric is
PFER=∑i​∣ui∗​∣∑i​D(feat(ui∗​),feat(ui​))​.
Here ui∗​ is the reference sequence, ui​ the predicted sequence, and D is feature-level edit distance.
The rationale is particularly appropriate for a benchmark of phone realization: a model that predicts an almost-correct segment should not necessarily be treated identically to one that predicts a completely unrelated articulation. 2601.14046v2
Lower PFER is better.

4. Extrinsic evaluation: transcription versus representation
The downstream portion is the most distinctive part of PRiSM.
For most downstream tasks they evaluate each model in two separate ways.
Transcript Probe (TP)
The PR model first generates a phone transcription. That phone sequence is fed to a lightweight bidirectional GRU trained for the downstream task:
x→PR→IPA sequence→BiGRU→y.
This asks:

How useful is the explicit phonetic transcription produced by this recognizer?

Representation Probe (RP)
Instead of decoding phones, the probe consumes the PR model's hidden representations. They use the final hidden layer, temporal attention pooling, and an MLP:
x→PR encoder→H→attention pooling→MLP→y.
This asks:

How useful is the information retained internally by the speech model?

They explicitly caution against interpreting TP and RP scores as interchangeable. The inputs contain different amounts and kinds of information, so meaningful comparisons are primarily TP-to-TP and RP-to-RP. They additionally test learned combinations of layers and report that the general RP trends remain. 2601.14046v2

5. Downstream tasks
The nine tasks fall into three groups.
Pathological speech


DYS-ez: dysarthria intelligibility prediction, EasyCall, Italian;


DYS-ua: dysarthria intelligibility prediction, UASpeech, English;


CSD-us: child speech-disorder detection, UltraSuite.


Metrics are rank correlation for intelligibility and F1 for disorder classification.
L2 speech


L1-eda: infer a speaker's L1 from English speech, EdAcc;


L1-arc: L1/accent classification from ARCTIC/L2-ARCTIC;


L2-so: pronunciation assessment, Speechocean762.


These test whether the recognizer captures pronunciation differences made by non-native speakers.
Multilingual/dialectal speech


LID-fl: language identification, FLEURS-24;


GEO-v: geolocation of Hindi dialect speech, Vaani;


PI-drc: infer the phone inventory of an unseen language, DoReCo.


These are deliberately heterogeneous. The benchmark is testing not merely whether one particular classifier can exploit phones, but whether the phone recognizer provides useful information across substantially different forms of phonetic variation. 2601.14046v2

6. Systems compared
The model comparison is structured so that architecture, pretraining and multilingual coverage vary.
Wav2Vec2-based phone recognizers
Three models are based on wav2vec-style SSL encoders and CTC:


W2V2P-LV60


W2V2P-XLSR53


MultiIPA


The important contrast is language exposure. MultiIPA uses multilingual XLSR pretraining but is phone-recognition-fine-tuned on only seven languages; the other systems have considerably broader PR training.
ZIPA


ZIPA-CTC


ZIPA-CTC-NS


These are Zipformer encoder systems trained from scratch with consistency-regularized CTC on IPAPack++. The basic training set covers 88 languages.
ZIPA-CTC-NS receives a further large pseudo-labelled stage covering around 4,000 languages.
POWSM


POWSM


POWSM-CTC


POWSM uses an E-Branchformer encoder plus Transformer decoder and hybrid CTC/attention training. POWSM-CTC is an encoder/CTC variant constructed by the authors.
This provides a relatively controlled test of autoregressive decoder dependencies versus CTC-style output.
Large Audio Language Models


Gemini 2.5 Flash


Qwen3-Omni-Instruct


These are evaluated principally through prompting because suitable aligned internal representations are either unavailable or difficult to extract.
Representation baselines
They also evaluate representations from:


WavLM-base


Whisper-small


even though these are not being used as PR transcription systems in that comparison. 2601.14046v2

7. Intrinsic phone-recognition results
There is no single system that dominates every dataset, but several broad patterns emerge.
For pronunciation variation within English, the best averages are around 10–11 PFER:
ModelSeen-language variation avg. PFERZIPA-CTC10.6ZIPA-CTC-NS10.6W2V2P-XLSR5310.8POWSM-CTC11.1W2V2P-LV6011.2Gemini 2.5 Flash13.7MultiIPA15.2POWSM17.5
The authors interpret the MultiIPA result as particularly informative. Its encoder received multilingual pretraining that included English, but English was not part of its phone-recognition fine-tuning. It performs noticeably worse than systems explicitly exposed to English PR data.
So multilingual SSL pretraining alone does not guarantee good phonetic transcription of a language.
POWSM also fails badly on Speech Accent Archive, with 27.6 PFER; the authors attribute this largely to decoder search problems on long speech sequences. Intriguingly, they note that a text-only G2P model reaches 10.2 PFER there—better than Gemini—even though a G2P system can only supply canonical pronunciation rather than actually perceive the speaker's realization. 2601.14046v2

Unseen-language results
The rankings change:
ModelUnseen-language avg. PFERPOWSM18.7ZIPA-CTC-NS19.0W2V2P-LV6019.5ZIPA-CTC19.6W2V2P-XLSR5321.0MultiIPA21.3POWSM-CTC21.9Gemini53.8Qwen3-Omni105.4
POWSM's advantage here is interesting because the encoder-decoder model was relatively poor on within-English variation. The authors argue that a degree of learned phone-sequence/phonological modelling can be helpful when generalizing to unfamiliar languages, even though the same reliance on likely sequences can hurt faithful realization transcription.
The headline conclusion is therefore not simply “CTC is better”:


when speech belongs to a known language, acoustic faithfulness and appropriate supervised language exposure matter strongly;


for a new language, broad multilingual exposure and learned cross-linguistic phonological structure can help.


The LALMs perform much worse on the unseen-language tests, although the very high averages partly arise from occasional catastrophic generations—long repetitions or insertions—which PFER penalizes heavily. 2601.14046v2

8. Extrinsic results: the best transcript is not necessarily the best representation
This is arguably the paper's most important empirical result.
Intrinsic PFER does not produce a stable ranking that predicts downstream usefulness.
The transcript probes and representation probes also rank systems quite differently.
For transcript probing, ZIPA systems and W2V2P-XLSR53 are generally strong. No system dominates all tasks. ZIPA-CTC-NS, for example, scores extremely strongly on child speech-disorder detection and multilingual language identification, while W2V2P-XLSR53 does well on several L2 and inventory tasks.
For representation probing, Whisper-small is the strongest system overall, with an aggregate score of 68.5, despite not being a specialized phonetic transcription model. Among the PR systems, ZIPA-CTC and ZIPA-CTC-NS are next at roughly 63. 2601.14046v2
This provides strong evidence for the benchmark's motivation:

The quality of a model's decoded IPA output and the richness of the phonetic information inside its encoder are genuinely different properties.


9. Different tasks prefer different information bottlenecks
The paper finds a systematic difference by application type.
Pathological speech tends to favour representations
RP tends to perform very well here. The authors suggest that clinical speech tasks may depend on properties such as:


timbre,


prosody,


fine acoustic realization,


which are discarded when continuous speech is reduced to an IPA sequence.
Multilingual/dialect tasks often favour transcription
Here, TP becomes relatively competitive or superior because language identity and dialect can be strongly encoded in:


phone inventories,


phone frequencies,


phonotactics,


phone-sequence distributions.


A discrete phone sequence may therefore function as a productive structured bottleneck, suppressing nuisance acoustic variation while preserving the linguistically useful differences.
L2 tasks occupy an intermediate position
Both explicit realization and richer acoustic information can matter.
This task-dependent TP/RP behaviour is one of the main reasons the authors argue against reducing phonetic model evaluation to a single transcription score. 2601.14046v2

10. Analysis 1: is a phone recognizer listening to the acoustics or completing likely phonotactics?
The paper then directly investigates a fundamental problem with PR.
A phone recognizer can output plausible phone sequences for two quite different reasons:
actual acoustic evidence
or
learned probability of phone sequences.
For genuine phonetic analysis, the former is normally what is wanted. A recognizer that “corrects” an unusual realization to the expected pronunciation may score well as ASR-like technology but fail as a measurement instrument.
Phone-masking experiment
Using TIMIT's aligned phone intervals, they replace an increasing percentage of phones in the waveform with silence.
They remove those same phones from the reference transcription and ask the recognizer to transcribe what remains.
If predictions depended solely on actual audible phones, increasing the masking rate should make little difference: the missing phones are no longer in either signal or reference.
Instead, several systems increasingly hallucinate phones as more context disappears.
Figure 2 on page 6 shows a striking architectural split. Wav2Vec2Ph and POWSM-CTC degrade relatively slowly, whereas POWSM and ZIPA degrade sharply at high masking levels. 2601.14046v2

Why this is not simply “CTC versus decoder”
POWSM's behaviour is intuitive: its autoregressive decoder conditions each prediction on previous predictions, so it has a direct mechanism for using phonotactics and for propagating errors.
POWSM-CTC lacks that dependency and remains substantially more acoustically grounded.
But ZIPA is also encoder-only and nevertheless deteriorates strongly.
The authors trace this to its consistency-regularized CTC (CR-CTC) training. CR-CTC explicitly encourages similar representations under noisy input. That is useful for robust recognition, but for fine phonetic measurement it can encourage the system to reconstruct expected phones despite missing acoustic evidence.
Insertion rates show the same trend: POWSM and ZIPA can output phones where the corresponding audio has literally been removed. 2601.14046v2
This gives an important qualification to the paper's broader praise of encoder-CTC systems:

architecture alone does not guarantee acoustic fidelity; the training objective matters as well.


11. The robustness–fidelity trade-off
This masking experiment leads to a broader interpretation of the benchmark results.
Some of the models that use learned phonological regularities most heavily do well on unseen languages, because a model can infer plausible segment structure from incomplete or unfamiliar acoustic evidence.
But that same tendency can hurt tasks where the objective is to preserve atypical realization, such as accented or pathological speech.
So “robustness” and “phonetic faithfulness” can actually conflict:
normalize / infer likely pronunciation↔transcribe what was physically produced.
That is an important conceptual point in PRiSM: the model that performs best as a general recognizer need not be the model one would trust most as a phonetic instrument.

12. Analysis 2: zero-shot phone-inventory induction
The DoReCo inventory task asks a particularly concrete linguistic question:

Given recordings from a previously unseen language, can a PR system recover the language's set of phones?

Each recognizer transcribes the DoReCo speech. The system's inventory is formed by taking the union of predicted phones and is compared with the reference inventory.
Most systems have the same basic pattern:


high recall, often above 70%;


poor precision, generally below 50%.


In other words, they recover many genuinely present phones but hallucinate many additional ones.
POWSM-CTC has the best precision/recall balance. ZIPA-CTC-NS is more precise than ZIPA-CTC, suggesting that the additional large multilingual pseudo-labelled training improves generalization to new inventories. 2601.14046v2
The authors draw an especially interesting data conclusion.
The Wav2Vec2 phone recognizers were trained on roughly 160,000 hours, compared with only about 17,000 hours for IPAPack++, but ZIPA's supervised data covers substantially more languages—88 rather than roughly 40.
ZIPA has better inventory recall.
They therefore argue that for universal PR,
language diversity can matter as much as raw hours​
and that multilingual coverage needs to occur not merely during SSL pretraining but also during supervised phone-recognition training. 2601.14046v2

13. Analysis 3: phone transcripts can encode surprisingly fine dialect geography
For Hindi dialect geolocation, transcript probes outperform representation probes substantially.
The paper reports approximately:


146 km mean error from transcription probes;


253 km from representation probes.


This is initially surprising because the transcription discards prosody and other continuous speech properties commonly associated with regional accent.
The authors argue that the phone sequence nevertheless preserves strong regional information through:


realization differences;


morphology;


phone frequencies;


sequence distributions.


There is also an architectural caveat: the TP is an RNN and therefore explicitly models sequential order, whereas the RP collapses time with attention pooling before an MLP. Some of TP's advantage may therefore result from the downstream probe itself rather than from transcription being intrinsically superior.
They illustrate the effect with Bangru/Haryanvi consonant doubling. In an example utterance, W2V2P-LV60 transcribes doubled [ll] and [kk], and integrated-gradient attribution shows that the geolocation classifier makes use of those phones. 2601.14046v2
This provides a concrete example of what PRiSM means by downstream utility: a phone recognizer may expose regional structure even when its transcription is imperfect globally.

14. Analysis 4: LALMs are poor fine-grained phonetic listeners
Gemini 2.5 Flash and Qwen3-Omni perform considerably worse than specialist systems overall.
The authors look closely at two zero-shot sociophonetic tasks.
Hindi dialect geolocation
Qwen predicts New Delhi for almost everything.
Gemini reaches only 6.5% hit@1, with around 65% of its outputs concentrated in only a few coordinate clusters near New Delhi.
The paper describes this as geographic mode collapse: the models favour a high-resource geographic prior rather than reliably discriminate the input speech.
L1/accent classification
The LALMs similarly overpredict a Romance-language accent cluster. Significant numbers of Slavic/Balkan and South Asian speakers are classified as Romance.
For Gemini, enabling reasoning/thinking makes performance worse, dropping F1 from 32.7 to 24.9. The reasoning traces frequently mention cues such as “Spanish/Italian/Portuguese” and “syllable-timed rhythm,” suggesting that the model constructs plausible verbal explanations around coarse stereotypes without actually resolving the relevant fine acoustic contrasts. 2601.14046v2
The authors therefore conclude that contemporary LALMs' general speech-language competence should not be mistaken for reliable fine-grained phonetic perception.
Few-shot examples improve some pathological-speech tasks but do not provide a consistent general fix. 2601.14046v2

15. What the benchmark ultimately finds
The paper's empirical conclusions can be reduced to four main points.
16. Transcription accuracy and phonetic usefulness are not equivalent
Systems move substantially in ranking between intrinsic PFER, transcript probes, and representation probes.
A meaningful evaluation of a PR system therefore needs both intrinsic and downstream tests.
17. Multilingual diversity matters throughout training
Broad multilingual SSL helps, but it is insufficient by itself. Broad supervised PR exposure substantially improves cross-lingual generalization, and diversity of languages can be more valuable than sheer audio quantity.
18. Encoder-CTC systems tend to provide stable phone recognition
They generally resist some kinds of sequence-model normalization and perform reliably across conditions. But stability also depends on the loss: ZIPA's consistency regularization demonstrates that an encoder-only architecture can still infer or hallucinate acoustically unsupported phones.
19. General-purpose LALMs do not replace specialized phone recognizers
Gemini and Qwen can produce IPA, but their intrinsic performance and fine sociophonetic discrimination remain markedly weaker, particularly cross-linguistically.
Whisper presents an interesting contrast: although not used here as a specialist phone decoder, its internal representations are exceptionally strong for the downstream probes.

20. What PRiSM actually measures
The benchmark is useful partly because it refuses to equate several quantities that are often conflated:
phone transcription accuracy=acoustic faithfulness=phonetic information in embeddings=downstream usefulness​
For example:


a system can achieve a plausible transcription by exploiting phonotactics rather than the signal;


a transcription can be imperfect while the encoder still contains excellent phonetic information;


a broad phone transcription can work extremely well for one downstream problem despite having thrown away fine acoustic detail;


a rich embedding can be worse for a task if the probe cannot exploit its structure;


modelling “likely” phones can improve unseen-language recognition while simultaneously destroying precisely the atypical pronunciations that a phonetician might want to study.


That distinction is the conceptual core of the paper.

Limitations
The authors emphasize several important restrictions.
First, phonetic transcription itself is not objective ground truth. It depends on the annotator, transcription conventions, inventory and desired level of detail. An IPA representation can also erase gradient or language-specific phenomena.
Second, the datasets necessarily provide incomplete coverage of languages, dialects, accents and speaking styles.
Third, both probe types introduce confounds. Transcript probes may exploit spurious sequence properties such as length, particularly when recognition quality is poor. Representation probing depends on choices such as layer selection and temporal pooling; the authors show that learned layer aggregation preserves the broad findings, but it remains a modelling choice.
Finally, they deliberately use mostly default decoding configurations and simple prompts rather than tuning each system for every benchmark task. The paper therefore claims a standardized comparative evaluation, not the maximum attainable score for every individual model. 2601.14046v2
Bottom line
PRiSM's central contribution is not a new phone recognizer but a framework for asking a more demanding question of one:

Does this system actually preserve and expose the phonetic realization in the speech signal, and is that information useful once we leave the transcription benchmark?

Its experiments show that those questions cannot be answered from PER/PFER alone. Language coverage, architecture, loss, decoding, phonotactic modelling and the distinction between decoded phones and hidden representations all materially change what a “good phone recognizer” means.


