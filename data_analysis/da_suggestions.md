

Emphasis on calls that the IVR could not handle
> 45% of IVR call types are payments and we handle those well. 
> Additionally, several callers get the needed details in our account summary (aka default balance), and those are handled well. 
> It's everything else that is interesting and puzzling.
 
what data patterns did you find interesting

it is important to know differences in the attribute correlation to "resolved" call types.

a good pitch combines data, storytelling, and awareness of the business goals

36 month delinquency history for the account. First byte is the current due stage at cycle, second byte is the prior due stage at cycle. 0 indicates current, 1 – one payment past due, 2 – two payments past due.

Suggestions for DA team:
 seperate observation and conclusions
 -for observation, write code to verify if it is statistical true : for example, if you observed TR at mos end will result in a floor call, did you check all the datas and verify this?
 -for conclusions, write in a seperate presentable file (maybe jupyter notebook/markdown file/txt or word file) to clearly state your findings, make sure use observations to support your statement  