# -*- coding: utf-8 -*-
"""
Created on Sun Jan  18 13:33:56 2021

@author: grundch
"""


import PyPDF2 as PDF
import os

DIR = os.getcwd()




def pdf_splitter(path, pages):
    fname = os.path.splitext(os.path.basename(path))[0]
    pdf = PDF.PdfFileReader(path)
    pdf_writer = PDF.PdfFileWriter()
    for page in range(pages[0]-1, pages[1]):
        pdf_writer.addPage(pdf.getPage(page))
    output_filename = '{}_pages_{} to {}.pdf'.format(fname, pages[0], pages[1])
    with open(output_filename, 'wb') as out:
            pdf_writer.write(out)
    print('Created: {}'.format(output_filename))

if __name__ == '__main__':
    path = 'PDF_Name.pdf'
    pages =[4,20]
    pdf_splitter(path, pages)

