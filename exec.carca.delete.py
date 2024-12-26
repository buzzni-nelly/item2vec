import argparse
import pathlib
from item2vec.volume import Volume


def clear_directory(directory_path: pathlib.Path):
    """재귀적으로 디렉토리 내 모든 파일 및 하위 디렉토리를 삭제합니다."""
    for x in directory_path.iterdir():
        if x.is_file():
            x.unlink()
        elif x.is_dir():
            clear_directory(x)
    directory_path.rmdir()


def main(company_id: str, version: str):
    model = "carca"
    volume_c = Volume(company_id=company_id, model=model, version=version)

    workspace_path = volume_c.workspace_path
    if workspace_path.exists() and workspace_path.is_dir():
        for item in workspace_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                clear_directory(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volume 작업을 실행합니다.")
    parser.add_argument(
        "--company-id",
        type=str,
        required=True,
        help="회사 ID를 입력하세요."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="버전 정보를 입력하세요."
    )
    args = parser.parse_args()

    main(company_id=args.company_id, version=args.version)
