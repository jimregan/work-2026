import click
from tokenizer import split_sentences
from embedder import embed, similarity_matrix
from align import align
from exporter import to_json


@click.command()
@click.argument("edition_a", type=click.Path(exists=True))
@click.argument("edition_b", type=click.Path(exists=True))
@click.option("--lang",      default="en",               help="Language for MosesSentenceSplitter")
@click.option("--model",     default="all-MiniLM-L6-v2", help="Sentence Transformers model name")
@click.option("--penalty",   default=0.1,  type=float,   help="Contiguity penalty weight per skipped sentence")
@click.option("--threshold", default=0.2,  type=float,   help="Minimum cosine similarity to form a match")
@click.option("--band",      default=10,   type=int,     help="DP band width")
@click.option("--output",    default="alignment.json",   help="Output file path")
def main(edition_a, edition_b, lang, model, penalty, threshold, band, output):
    click.echo("Splitting sentences...")
    sents_a = split_sentences(open(edition_a, encoding="utf-8").read(), lang)
    sents_b = split_sentences(open(edition_b, encoding="utf-8").read(), lang)
    click.echo(f"  Edition A: {len(sents_a)} sentences")
    click.echo(f"  Edition B: {len(sents_b)} sentences")

    click.echo(f"Embedding with '{model}'...")
    emb_a = embed(sents_a, model)
    emb_b = embed(sents_b, model)
    sim = similarity_matrix(emb_a, emb_b)

    click.echo(f"Aligning (penalty={penalty}, threshold={threshold}, band={band})...")
    pairs = align(sim, penalty_weight=penalty, null_threshold=threshold, band=band)

    meta = {
        "edition_a": edition_a,
        "edition_b": edition_b,
        "model": model,
        "penalty_weight": penalty,
        "null_threshold": threshold,
        "band": band,
    }
    to_json(output, sents_a, sents_b, pairs, meta)
    click.echo(f"Done. {len(pairs)} rows written to '{output}'.")


if __name__ == "__main__":
    main()
