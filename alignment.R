library(tidyverse)
library(here)
library(glue)
library(lubridate)

ts_loc <- TRANSCRIPT_LOC
# transcripts live in here(ts_loc, "csv"), with subfolders wer participant
# diarisations live in here(ts_loc, "output_voice_type_classifier"), with subfolders wer participant

ts_ppts <- list.dirs(here(ts_loc, "csv"))[-1]

for (ppt in ts_ppts) {
  ts_files <- list.files(ppt, pattern = "*.csv", recursive = TRUE)
  diar <- read_delim(here(str_replace(ppt, "csv", "output_voice_type_classifier"),
                          "all.rttm"),
                     col_names = c("V1", "filename", "V3", "time_start", "dur",
                                   "V6", "V7", "speaker", "V9", "V10"),
                     show_col_types = FALSE) |>
    select(-starts_with("V")) |>
    filter(speaker != "SPEECH") |>
    mutate(time_end = time_start + dur)

  out_loc <- here(str_replace(ppt, "csv", "diarised"))
  dir.create(out_loc, showWarnings = FALSE, recursive = TRUE)

  for (ts_file in ts_files) {
    ts <- read_csv(here(ppt, ts_file),
                   show_col_types = FALSE)
    
    ts_utt <- ts |> 
      group_by(utterance_id, utterance) |> 
      summarise(start_time = min(token_start_time), end_time = max(token_end_time), .groups = "drop")

    cur_diar <- diar |>
      filter(filename == tools::file_path_sans_ext(basename(ts_file)))

    if (nrow(cur_diar) == 0 | nrow(ts) == 0) next

    ts_diar <- ts_utt |>
      left_join(cur_diar, by = join_by(overlaps(start_time, end_time, time_start, time_end))) |>
      mutate(dur_incl = pmin(end_time, time_end) - pmax(start_time, time_start)) |>
      nest(diar = -all_of(colnames(ts_utt))) |>
      mutate(speaker = sapply(diar, \(d) {
        d |>
          group_by(speaker) |>
          summarise(dur_incl = sum(dur_incl, na.rm = TRUE)) |>
          arrange(desc(dur_incl)) |>
          slice(1) |>
          pull(speaker)
      })) |>
      select(-diar)
    
    ts_full <- ts |> 
      left_join(ts_diar |> select(-start_time, -end_time), by = join_by(utterance_id, utterance)) |>
      mutate(speaker = ifelse(is.na(speaker), "unknown", speaker),
             token_start_time = token_start_time |> 
               seconds_to_period() |> 
               as_datetime() |> 
               format("%H:%M:%S"),
             token_end_time = token_end_time |> 
               seconds_to_period() |> 
               as_datetime() |> 
               format("%H:%M:%S"))

    write_csv(ts_full, here(out_loc, ts_file |> str_remove("_processed")))
  }
}
