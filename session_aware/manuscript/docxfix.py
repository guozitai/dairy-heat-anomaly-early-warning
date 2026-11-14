#!/usr/bin/env python3
"""Post-process a docx written by the converter: full-width tables, fonts that exist in Word,
page numbers in the footer and, for the manuscript, continuous line numbering."""
import re, zipfile, shutil
FTR=('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
     '<w:p><w:pPr><w:suppressLineNumbers/><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:ind w:left="0" w:right="0" w:firstLine="0"/><w:jc w:val="center"/></w:pPr>'
     '<w:r><w:rPr><w:rFonts w:ascii="FONT" w:hAnsi="FONT"/><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
     '<w:r><w:rPr><w:rFonts w:ascii="FONT" w:hAnsi="FONT"/><w:sz w:val="20"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
     '<w:r><w:rPr><w:rFonts w:ascii="FONT" w:hAnsi="FONT"/><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
     '<w:r><w:rPr><w:rFonts w:ascii="FONT" w:hAnsi="FONT"/><w:sz w:val="20"/></w:rPr><w:t>1</w:t></w:r>'
     '<w:r><w:rPr><w:rFonts w:ascii="FONT" w:hAnsi="FONT"/><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
TW=[]  # column widths (per cent) registered by table() in document order; None where the converter's proportions are kept
def scale_tables(t, widths=None, pad=70):
    """Set the columns of every table to the text width (registered proportions where given), give the cells a
    horizontal margin, keep rows and captions together and repeat the header row."""
    pg=re.search(r'<w:pgSz w:w="(\d+)"',t); mg=re.search(r'<w:pgMar [^>]*?w:left="(\d+)" w:right="(\d+)"',t)
    if not (pg and mg): return t
    target=int(pg.group(1))-int(mg.group(1))-int(mg.group(2)); n=len(re.findall(r'<w:tbl>',t)); use=widths if (widths is not None and len(widths)==n) else None; k=[0]
    def one(m):
        x=m.group(0); g=[int(v) for v in re.findall(r'<w:gridCol w:w="(\d+)"/>',x)]; w=use[k[0]] if use else None; k[0]+=1
        if not g or sum(g)==0 or '<w:gridSpan' in x: return x
        ng=[round(target*v/sum(w)) for v in w] if (w and len(w)==len(g)) else [round(v*target/sum(g)) for v in g]
        it=iter(ng); x=re.sub(r'<w:gridCol w:w="\d+"/>',lambda _:f'<w:gridCol w:w="{next(it)}"/>',x)
        def row(rm):
            c=iter(ng); return re.sub(r'<w:tcW w:w="\d+" w:type="dxa"/>',lambda _:f'<w:tcW w:w="{next(c)}" w:type="dxa"/>',rm.group(0))
        x=re.sub(r'<w:tr>.*?</w:tr>',row,x,flags=re.S)
        x=x.replace('<w:trPr></w:trPr>','<w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>',1).replace('<w:trPr></w:trPr>','<w:trPr><w:cantSplit/></w:trPr>')
        x=re.sub(r'(<w:tblCellMar>.*?)<w:left w:w="\d+" w:type="dxa"/>(.*?)<w:right w:w="\d+" w:type="dxa"/>(.*?</w:tblCellMar>)',rf'\1<w:left w:w="{pad}" w:type="dxa"/>\2<w:right w:w="{pad}" w:type="dxa"/>\3',x,count=1,flags=re.S)
        return x
    t=re.sub(r'<w:tbl>.*?</w:tbl>',one,t,flags=re.S)
    parts=t.split('<w:tbl>')  # keep the paragraph before each table (its caption) on the page of the table
    for q in range(len(parts)-1):
        p=parts[q].rfind('<w:p>'); h=parts[q][p:]
        if p<0 or '<w:keepNext/>' in h: continue
        h=re.sub(r'(<w:pPr>(?:<w:pStyle [^>]*/>)?)',r'\1<w:keepNext/>',h,count=1) if '<w:pPr>' in h else h.replace('<w:p>','<w:p><w:pPr><w:keepNext/></w:pPr>',1)
        parts[q]=parts[q][:p]+h
    return '<w:tbl>'.join(parts)
def fix_docx(path, latin="Times New Roman", cjk="SimSun", line_numbers=False, page_numbers=True, widths=None):
    zi=zipfile.ZipFile(path); items=[(i,zi.read(i.filename)) for i in zi.infolist()]; zi.close()
    zo=zipfile.ZipFile(path+".tmp","w",zipfile.ZIP_DEFLATED)
    for i,d in items:
        if i.filename.startswith("word/") and i.filename.endswith(".xml"):
            t=d.decode("utf-8")
            if i.filename=="word/document.xml":
                t=re.sub(r'<w:tblW w:w="\d+" w:type="dxa"/>','<w:tblW w:w="5000" w:type="pct"/>',t)
                t=scale_tables(t,widths)
                if line_numbers: t=re.sub(r'(<w:pgMar [^>]*/>)',r'\1<w:lnNumType w:countBy="1" w:restart="continuous"/>',t)
                if page_numbers:
                    assert 'xmlns:r=' in t[:3000]
                    t=t.replace('<w:sectPr>','<w:sectPr><w:footerReference w:type="default" r:id="rIdFtr1"/>'); t=re.sub(r'(<w:pgMar [^>]*?)w:footer="\d+"',r'\1w:footer="567"',t)
            for a in ("Times New Roman;serif","SimSun;宋体;Noto Serif CJK SC;serif","Liberation Sans Unicode MS","Liberation Sans","Liberation Serif"): t=t.replace('"'+a+'"','"'+(cjk if a.startswith("SimSun") else latin)+'"')
            t=t.replace('"Noto Serif CJK SC"','"'+cjk+'"'); d=t.encode("utf-8")
        elif page_numbers and i.filename=="word/_rels/document.xml.rels":
            d=d.decode("utf-8").replace("</Relationships>",'<Relationship Id="rIdFtr1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>').encode("utf-8")
        elif page_numbers and i.filename=="[Content_Types].xml":
            d=d.decode("utf-8").replace("</Types>",'<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>').encode("utf-8")
        zo.writestr(i,d)
    if page_numbers: zo.writestr("word/footer1.xml",FTR.replace("FONT",latin))
    zo.close(); shutil.move(path+".tmp",path)
