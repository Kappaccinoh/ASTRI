# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import re
from typing import List, Optional

import fire

from llama import Dialog, Llama

import requests

import xml.etree.ElementTree as ET

from addressToGPS import addressToGPS

from nearestCam import distance_to_cam


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # API endpoint
    url = "https://www.td.gov.hk/datagovhk_tis/traffic-notices/Notices_on_Clearways.xml"
    # url = "https://www.td.gov.hk/datagovhk_tis/traffic-notices/Notices_on_Temporary_Road_Closure.xml"
    # url = "https://www.td.gov.hk/datagovhk_tis/traffic-notices/Notices_on_Temporary_Speed_Limits.xml"
    # url = "https://www.td.gov.hk/datagovhk_tis/traffic-notices/Notices_on_Prohibited_Zone.xml"

    # Send a GET request to the API
    response = requests.get(url)

    # Parse the XML response
    root = ET.fromstring(response.content)

    # Find all the <Notice> elements
    notices = root.findall("Notice")

    notice_number = 100
    curr = 0
    correct_output = 0

    # Write the <Content_TC> content to a text file
    
    for notice in notices:
        if curr < notice_number:
            with open("content_tc.txt", "w", encoding="utf-8") as f:
                content_tc = notice.find("Content_TC").text
                content_en = notice.find("Content_EN").text
                f.write(content_en)
                f.write("---\n")
                f.write(content_tc)
                f.write("---\n")
                f.close()
                # Append the content from prompt_requirements.txt
                with open("prompt_requirements.txt", "r", encoding="utf-8") as p, \
                    open("content_tc.txt", "a", encoding="utf-8") as f:
                    prompt_content = p.read()
                    f.write("\n\n" + prompt_content)

                file = open('content_tc.txt', 'r')
                prompt = file.read()

                dialogs: List[Dialog] = [
                    [{"role": "user", "content": prompt}]
                ]
                results = generator.chat_completion(
                    dialogs,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                # Extract result from llama3 and feed it into the GPS searcher
                output = results[0]['generation']['content']
                print("------------------------------------\n")
                print(output)
                print("------------------------------------\n")


                # streets = output.split(",")

                # if len(streets) > 1:
                #     street1 = streets[0].strip()
                #     street2 = streets[1].strip()
                #     returned_coords = addressToGPS(street1, street2)
                #     if returned_coords == (1,0):
                #         street1 = streets[int(len(streets)/2)].strip()
                #         print("TRIED ENG")
                #         returned_coords = addressToGPS(street1, street2)
                #     if returned_coords == (0,1):
                #         try:
                #             street2 = streets[int(len(streets)/2) + 1].strip()
                #             returned_coords = addressToGPS(street1, street2)
                #         except:
                #             returned_coords = (0,0)
                # else:
                #     returned_coords = (0,0)
                # if returned_coords != (0,0):
                #     correct_output += 1
                # print("Number of correct outputs so far:")
                # print(f"{correct_output}/{curr + 1}")
                # distance_to_cam(returned_coords)
            curr += 1
        else:
            break
        

    print("Total number of correct outputs:")
    print(f"{correct_output}/{notice_number}")
    # Print out result to terminal
    # for dialog, result in zip(dialogs, results):
    #     for msg in dialog:
    #         print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    #     print(
    #         f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    #     )
    #     print("\n==================================\n")




if __name__ == "__main__":
    fire.Fire(main)
