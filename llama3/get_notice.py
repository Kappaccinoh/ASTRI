import pandas as pd
import xml.etree.ElementTree as ET


df = pd.read_excel('Datasets/Clearways_Dataset_24_07.xlsx', sheet_name= 'EN_CN_New_Format')
df = df.iloc[0:, 0]

with open('Notices_on_Clearways.xml', 'rb') as file:
    xml_data = file.read()
    root = ET.fromstring(xml_data)
    notices = []
    for notice in root.findall(".//Notice")[:100]:
        content_en = notice.find("Content_EN").text
        content_tc = notice.find("Content_TC").text + content_en
        # print(content_tc)
        # notices.append(content_en)
        notices.append(content_tc) 
new_df = pd.DataFrame({"Notice": notices})

df = pd.concat([df, new_df], axis=1)
df.to_excel("Datasets/Clearways_train_EN_CN_New_Format.xlsx", index=False)

print(df)
