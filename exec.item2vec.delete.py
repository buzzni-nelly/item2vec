import argparse

from item2vec.volume import Volume


def delete_checkpoints(company_id: str, version: str):
    model = "item2vec"  # 모델 이름 고정
    volume = Volume(company_id=company_id, model=model, version=version)
    print(f"Removing checkpoints from {volume.checkpoints_dirpath}...")
    for file in volume.checkpoints_dirpath.glob("*"):
        if file.is_file():
            file.unlink()


def main(company_id: str, version: str):
    delete_checkpoints(company_id, version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkpoints 삭제 스크립트")
    parser.add_argument(
        "--company-id",
        type=str,
        required=True,
        help="처리할 회사 ID를 입력하세요."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="모델 버전을 입력하세요."
    )
    args = parser.parse_args()

    main(args.company_id, args.version)
