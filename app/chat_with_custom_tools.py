# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import mesop as me

from examples.custom_tools.ticker_data import TickerDataTool
from llama_agentic_system.tools.genes import GeneDiseaseAssociationTool, GeneGoTermsTool, DownstreamAnalysisTool
from utils.chat import chat, State
from utils.client import ClientManager
from utils.common import DISABLE_SAFETY, INFERENCE_HOST, INFERENCE_PORT, on_attach
from utils.transform import transform


client_manager = ClientManager()
client_manager.init_client(
    inference_port=INFERENCE_PORT,
    host=INFERENCE_HOST,
    custom_tools=[TickerDataTool(), GeneDiseaseAssociationTool(), GeneGoTermsTool(),DownstreamAnalysisTool()],
    disable_safety=DISABLE_SAFETY,
)


@me.page(
    path="/",
    title="Llama Agentic System",
)
def page():
    state = me.state(State)
    chat(
        transform,
        title="Llama Agentic System",
        bot_user="Llama Agent",
        on_attach=on_attach,
    )


if __name__ == "__main__":
    import subprocess

    subprocess.run(["mesop", __file__])
