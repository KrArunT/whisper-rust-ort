# optional: start fresh
# rm -f RESULTS.md RESULTS.csv

find results/benchmarks -type f -name summary_table.md -path '*/container_*/*' | while read -r table; do
  sut_dir="$(dirname "$table")"
  parent="$(basename "$sut_dir")"
  if [[ "$parent" == container_* ]]; then
    sut_name="default"
    container_name="$parent"
  else
    sut_name="$parent"
    container_name="$(basename "$(dirname "$sut_dir")")"
  fi

  if [[ "$container_name" =~ ^container_([0-9]+)c([0-9]+)g$ ]]; then
    cores="${BASH_REMATCH[1]}"
    mem="${BASH_REMATCH[2]}"
  else
    echo "Skip: unexpected path $container_name"
    continue
  fi

  summary_csv="${table%.md}.csv"
  uv run update_results_md.py \
    --results-md RESULTS.md \
    --summary-table "$table" \
    --summary-csv "$summary_csv" \
    --sut-name "$sut_name" \
    --core-count "$cores" \
    --memory-gb "$mem" \
    --results-csv RESULTS.csv
done

