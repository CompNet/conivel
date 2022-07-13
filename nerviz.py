import os
from typing import Optional, List
import torch
from IPython.display import display, HTML, Javascript
import ipywidgets as widgets
from conivel.datas import NERSentence
from conivel.utils import flattened


_script_dir = os.path.dirname(os.path.abspath(__file__))


def _refresh_attention(attentions: torch.Tensor):
    """
    :param attentions: ``(sentence_size, sentence_size)``
    """
    display(
        Javascript(
            f"window.attentions = {attentions.tolist()}; window.refreshAttentions();"
        )
    )


def visualise_ner_sent_attentions(
    sent: NERSentence,
    attentions: torch.Tensor,
    pred_tags: Optional[List[str]] = None,
    add_special_tokens: bool = False,
):
    """Interactively visualize attentions for a NER sentence

    :param sent:
    :param attentions: a tensor of shape ``(layers_nb, heads_nb,
        sentence_size, sentence_size)``
    :param pred_tags: if given, tag predictions for ``sent``
    :param add_special_tokens: if ``True``, add special tokens as in
        :func:`NERSentence.flattened`.
    """
    layers_nb = attentions.shape[0]
    heads_nb = attentions.shape[1]

    tokens, tags = sent.flattened(add_special_tokens=add_special_tokens)

    tokens_html = []
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        token_source_html = (
            f"<div class='token source' style='display: table-cell;'>{token}</div>"
        )
        token_target_html = (
            f"<div class='token target' style='display: table-cell;'>{token}</div>"
        )
        tag_html = f"<div style='display: table-cell; width: 10%;'>{tag}</div>"
        if not pred_tags is None:
            pred_tag = pred_tags[i]
            color = "green" if pred_tag == tag else "red"
            tag_html += f"<div style='display: table-cell; width: 10%; color: {color}'>{pred_tag}</div>"
        tokens_html += "<div style='display: table-row;'> {} {} {} </div>".format(
            tag_html, token_source_html, token_target_html
        )
    tokens_html = "".join(tokens_html)

    viz_html = HTML(
        "<div id='viz' style='display: table; width: 100%'>\n{}</div>".format(
            tokens_html
        )
    )

    viz_js = Javascript(filename=f"{_script_dir}/nerviz.js")

    # options
    layers_dropdown = widgets.Dropdown(
        description="layer",
        options=[str(i) for i in range(layers_nb)],
        value=str(layers_nb - 1),
    )
    heads_dropdown = widgets.Dropdown(
        description="head",
        options=[str(i) for i in range(heads_nb)],
        value=str(heads_nb - 1),
    )

    def on_options_change(event):
        layer = int(layers_dropdown.value)  # type: ignore
        head = int(heads_dropdown.value)  # type: ignore
        _refresh_attention(attentions[layer][head])

    layers_dropdown.observe(on_options_change, names="value")
    heads_dropdown.observe(on_options_change, names="value")

    options = widgets.HBox([layers_dropdown, heads_dropdown])

    display(options, viz_html, viz_js)
    on_options_change(None)
