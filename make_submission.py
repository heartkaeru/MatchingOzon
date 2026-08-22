"""
Скрипт для упаковки файлов решения в архив submission.zip для отправки на платформу.
"""
import zipfile
import os
import shutil


def make_archive():
    submission_dir = "submission"
    src_dir = "src"
    target_src_in_submission = os.path.join(submission_dir, "src")

    # Синхронизация папки src в папку submission/src перед упаковкой
    if os.path.exists(target_src_in_submission):
        shutil.rmtree(target_src_in_submission)
    shutil.copytree(src_dir, target_src_in_submission)

    # Создание архива submission.zip
    zip_filename = "submission.zip"
    print(f"Создание архива {zip_filename}...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                if file.endswith(".gitkeep"):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(submission_dir))
                zipf.write(file_path, arcname)
    print("Архив успешно создан и готов к отправке!")


if __name__ == "__main__":
    make_archive()

