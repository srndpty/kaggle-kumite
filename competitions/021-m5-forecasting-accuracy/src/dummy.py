"""P0 用に公式サンプルと同じ形式の季節ナイーブ提出を作る。"""

from dataset import PRIVATE_ORIGIN, PUBLIC_ORIGIN, ROOT, load_sales, submission
from model import seasonal


def main() -> None:
    public = load_sales()
    private = load_sales(evaluation=True)
    csv = submission(
        public, seasonal(public, PUBLIC_ORIGIN), private, seasonal(private, PRIVATE_ORIGIN)
    )
    path = ROOT / "outputs" / "p0-dummy.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    csv.to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
