import os
import win32com.client as win32

# Checking whether files uploaded are of the expected types, if not raise an error
def check_files(files):
    unallowed_files = []
    allowed_extensions = ['.pdf', '.csv', '.docx', '.md', '.doc', '.txt']
    for file in files:
        file_ext = os.path.splitext(str(file))[1].lower()
        if file_ext not in allowed_extensions:
            unallowed_files.append(file)

    return unallowed_files
            
# To convert unsupported doc files, if present to docx files
def convert_files(source_directory):
    word = win32.gencache.EnsureDispatch('Word.Application')
    for filename in os.listdir(source_directory):
        if filename.endswith('.doc'):
            doc_path = os.path.join(source_directory, filename)
            docx_path = os.path.join(source_directory, filename.replace('.doc', '.docx'))
            doc = word.Documents.Open(doc_path)
            doc.SaveAs2(docx_path, FileFormat=16)
            doc.Close()
            # Move the original file to the output directory
            os.remove(doc_path)
    word.Quit()
    # Convert all the doc files to docx files in the same directory
    # ERRORS 404 because you would be trying to open the file
    pass

