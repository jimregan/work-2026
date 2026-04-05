One limitation of Sparrowhawk (as well as the implementation in NeMo text processing) is the lack of integration with external tools. In Kestral, a part of speech tagger is used to add case information, in languages where it is relevant, meaning that the correct case form of an NSW can be produced. For Northern Sami in particular, we rely on the presence of a case suffix to convey this information, though it seems relatively clear that this cannot always be depended on to be present.

In the case of audio-backed text normalisation, we can … all case forms of the NSW, allowing the acoustic evidence to be used in such cases.

(This can be overcome?)