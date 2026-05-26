
Classical TTS systems expose linguistic structure through text normalization, lexicons, G2P, phonological rules, and duration models. These components can be incomplete or over-specific, but their assumptions are inspectable and have benefited from decades of practical engineering.

Modern end-to-end systems move many of these decisions into latent representations. This can improve fluency and reduce pipeline complexity, but it often replaces explicit, correctable failure modes with opaque ones, especially for long-tail pronunciation, dialect, code-switching, names, and accessibility use cases.

We propose scheduled structural internalization: use explicit linguistic machinery as training-time scaffolding and diagnostic interface, while gradually encouraging the model to internalize the underlying structure in latent form.