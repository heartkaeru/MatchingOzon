"""
Utility script to package submission files into submission.zip
"""
import zipfile
import os
import shutil

def make_archive():
    submission_dir = "submission"
    src_dir = "src"
    target_src_in_submission = os.path.join(submission_dir, "src")

    # Sync src to submission/src before packing
    if os.path.exists(target_src_in_submission):
        shutil.rmtree(target_src_in_submission)
    shutil.copytree(src_dir, target_src_in_submission)

    # Create submission.zip
    zip_filename = "submission.zip"
    print(f"Creating {zip_filename}...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                if file.endswith(".gitkeep"):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(submission_dir))
                zipf.write(file_path, arcname)
    print("Done!")

if __name__ == "__main__":
    make_archive()
