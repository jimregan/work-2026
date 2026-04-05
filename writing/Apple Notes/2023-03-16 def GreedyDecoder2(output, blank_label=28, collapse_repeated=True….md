def GreedyDecoder2(output, blank_label=28, collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = \[\]
        for i, args in enumerate(arg_maxes):
                decode = \[\]
                for j, index in enumerate(args):
                        if index != blank_label:
                                if collapse_repeated and j != 0 and index == args\[j -1\]:
                                        continue
                                decode.append(index.item())
                decodes.append(text_transform.int_to_text(decode))
        return decodes