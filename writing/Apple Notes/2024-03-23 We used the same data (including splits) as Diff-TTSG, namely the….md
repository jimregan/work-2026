We used the same data (including splits) as Diff-TTSG, namely the Trinity Speech-Gesture Dataset II (TSGD2)1 \[35, 36\]. The dataset contains 6 hours of multimodal recordings (time-aligned 44.1 kHz audio and 120 fps marker-based motion-capture) of a male native speaker of Hiberno English, discussing various topics with free gestures. 1.5 hours of this data was held out for testing, with the rest used for training. We likewise used the same text transcriptions, acoustic features, pose representations (matching the frame rate of the acoustics), and HiFi-GAN vocoder2 \[37\] as \[11\] followed by a denoising filter \[38\] with strength 2e-4. We also used the same neutral-looking, skinned upper-body avatar to visualise motion, which omits finger motion (due to inaccurate finger mocap) as well as facial expression, gaze, and lip motion (which are not captured by the dataset nor synthesised by the methods considered).

Trinity Speech-Gesture Dataset II (TSGD2) 

In this work, we use the same evaluation protocol as \[11\], including the same questions asked, response options, participant recruitment, attention checks and inclusion criteria, as well as statistical analysis.



**Diff-TTSG** (first, longer paper with more details):
		Topics common to the setup of the entire evaluation
		- Segments selected
		- Vocoding
		- Common features of the subjective evaluations
		- Attention check design
		- Speech-only evaluation
		- Setup of unimodal speech user study
		- Results and interpretation of unimodal speech user study
		- Gesture-only evaluation
		- Setup of unimodal motion user study
		- Results and interpretation of unimodal motion user study
		- Speech-and-gesture evaluation
		- Detailed setup of cross-modal user study, and how it differs from prior work
		- Analysis method and what a score of 0 means
		- Results and interpretation of cross-modal user study
		- With extra detailed speculation on why Diff-TTSG was not as appropriate as the natural gestures, since a reviewer wanted that
**Match-TTSG** (shorter paper that refers back to Diff-TTSG and said we did it similarly – we can do the same where we want):
		Objective results (in the order setup, result, interpretation for each result)
		- Setup of unimodal user studies
		- Very brief since it basically says "In this work, we use the same evaluation protocol as Diff-TTSG", and reports the essential numbers/differences
		- Setup of cross-modal user study (including how responses are assigned numerical values)
		- Results (significant differences) of the unimodal studies
		- Discussion of the unimodal user studies
		- Easy since "The unimodal evaluations exhibited similar trends for both audio and video"
		- Results (significant differences) of the cross-modal study
		- Discussion of the cross-modal study
		- One-sentence summary of all results, prior to the conclusions section
		- Conclusions