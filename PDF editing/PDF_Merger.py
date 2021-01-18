# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:33:56 2021

@author: grundch
"""


import PyPDF2 as PDF
import itertools as itt
import os

DIR = os.getcwd()



pdfs = []

for filename in os.listdir(DIR):
    if filename.endswith(".pdf"):
        pdfs.append(filename)

pdf_out = PDF.PdfFileWriter()

with open(pdfs[1], 'rb') as f_odd:
    with open(pdfs[0], 'rb')  as f_even:
        pdf_odd = PDF.PdfFileReader(f_odd)
        pdf_even = PDF.PdfFileReader(f_even)

        for p in itt.chain.from_iterable(
            itt.zip_longest(
                pdf_odd.pages,
                pdf_even.pages,
            )
        ):
            if p:
                pdf_out.addPage(p)

        with open("all.pdf", 'wb') as f_out:
            pdf_out.write(f_out)


