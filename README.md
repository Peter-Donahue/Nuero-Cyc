# Nuero-Cyc

https://www.economist.com/obituary/2023/09/13/douglas-lenat-trained-computers-to-think-the-old-fashioned-way

I came across an [obituary in The Economist for Douglas Lenat]([url](https://www.economist.com/obituary/2023/09/13/douglas-lenat-trained-computers-to-think-the-old-fashioned-way)), with whom I was unfamiliar at the time. This lead me to look up Cyc and from there learn about the discontinued project OpenCyc. Inspired by the moment, I wanted to see if I could wrap it in an LLM and get it to do anything interesting. In its current state, this project is not very polished, I may or may not tinker with it more at some point.

The main idea is pretty simple: 
 - You send a message
 - An LLM:
  - Interprets the message, crafts CycL (the unambigous, somewhat tedious, langauge used by Cyc)
  - Confirms the Cyc Knowledge Base (KB) has the necessary bits of information needed to complete the required reasoning. If the KB is missing anything, the LLM supplies its best guess as a microtheorom (which is session-scoped and wont "pollute" the KB).
  - Sends the CycL message to Cyc
  - If Cyc throws an error the LLM will try to account for the issue and redo these steps. Current limited to 4 retries. 
 - Cyc will perform all the reasoning and produce a response.
 - The LLM will process that response before returning it to the user. Technically Cyc can prodcue some sort of natural language responses, but they seem a little redamentary, and I thought it would be helpful to have this second LLM interaction point to potentially allow for more flexibility in the format of the response (ie, if the user wanted it in LaTex)
