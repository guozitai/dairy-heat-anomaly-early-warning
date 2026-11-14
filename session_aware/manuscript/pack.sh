#!/bin/bash
# Assemble the resubmission package and zip it.
set -e
MS=/work/cea_sat/rescue/ms; R=/work/cea_sat/rescue; PK=$R/pkg_v2/SAT重投包-v2-20261111
rm -rf $R/pkg_v2; mkdir -p "$PK/01_上传件/figures" "$PK/01_上传件/pdf_预览" "$PK/02_源文件/text" "$PK/02_源文件/code/manuscript" "$PK/02_源文件/code/analysis" "$PK/02_源文件/results"
cp $MS/response.docx "$PK/01_上传件/01_Cover_Letter_and_Response.docx"; cp $MS/highlights.docx "$PK/01_上传件/02_Highlights.docx"; cp $MS/manuscript.docx "$PK/01_上传件/03_Manuscript.docx"; cp $MS/supplementary.docx "$PK/01_上传件/04_Supplementary_Material.docx"
cp $MS/declaration_of_interests.docx "$PK/01_上传件/05_Declaration_of_Interests.docx"
for i in 1 2 3 4 5; do cp $MS/figs/Fig$i.pdf $MS/figs/Fig$i.png "$PK/01_上传件/figures/"; done
mkdir -p "$PK/01_上传件/figures/上传用_顶部留白" && cp $MS/figs/upload/Fig*.pdf "$PK/01_上传件/figures/上传用_顶部留白/"
cp $MS/manuscript.pdf $MS/supplementary.pdf $MS/response.pdf "$PK/01_上传件/pdf_预览/"
cp $MS/投稿清单.docx "$PK/00_投稿清单.docx"
cp $MS/manuscript.md $MS/abstract.md $MS/results_a.md $MS/results_b.md $MS/discussion.md $MS/conclusions.md $MS/back.md $MS/response.md $MS/highlights.md $MS/manuscript_full.md $MS/*.tpl.md "$PK/02_源文件/text/"
cp $MS/build2.py $MS/build2_supp.py $MS/docxfix.py $MS/gen_results_b.py $MS/figstyle.py $MS/fig1.py $MS/fig2.py $MS/fig3.py $MS/fig4.py $MS/fig5.py $MS/figS.py $MS/mk_checklist.py $MS/pack.sh "$PK/02_源文件/code/manuscript/"
cp $R/v2/a1_build.py $R/v2/a1c_cowday.py $R/v2/a1d_lagref.py $R/v2/a2_weather.py $R/v2/b0_features.py $R/v2/b1_audit.py $R/v2/b3_herd.py $R/v2/b3x_herd_robust.py $R/v2/b3y_herd_final.py $R/v2/b3w_completeness.py $R/v2/b6_desc.py $R/v2/b8_misc.py $R/v2/b9_submitted.py $R/v2/c0_audit_extra.py $R/v2/c0b_missing_pattern.py $R/v2/c1_ew.py $R/v2/c2_regroup.py $R/v2/c3_flow.py $R/v2/c4_validity.py $R/v2/c5_cohort_label.py $R/v2/c5b_cohort_cows.py $R/v2/c6_merge.py $R/v2/c7_herd_extra.py $R/v2/c8_forecast_local.py $R/v2/c9_complete_days.py $R/v2/c10_lowyield.py $R/v2/c11_my24_stats.py $R/v2/b3z_checks.py $R/v2/a2b_elevation.py $R/v2/b8_records.py $R/build_my24.py $R/v2/d1_farm3_intervals.py $R/v2/d1b_farm3_fourth.py $R/v2/d2_boundary_lowyield.py $R/v2/d3_forecast_ordered.py $R/v2/d4_farm3_fourth_evidence.py $R/v2/d5_without_farm3_2025.py $R/v2/d6_without_farm3_local.py "$PK/02_源文件/code/analysis/"
cp $R/v2/out/*.csv $R/v2/out/*.log "$PK/02_源文件/results/" 2>/dev/null || true
rm -f "$PK/02_源文件/results/"*_unmatched* "$PK/02_源文件/results/"*_descriptive* "$PK/02_源文件/results/"*partial* "$PK/02_源文件/results/"b45_ew.log "$PK/02_源文件/results/"b5x_variants.log "$PK/02_源文件/results/"b7_sens.log "$PK/02_源文件/results/"ew_disease_variants.csv
cp $R/LIT_BRIEF.md $R/REVISION_TODO.md "$PK/02_源文件/"
mkdir -p "$PK/02_源文件/github" && cp $MS/github_handoff/session-aware.bundle $MS/github_handoff/说明.txt "$PK/02_源文件/github/"
cd $R/pkg_v2 && rm -f ../SAT重投包-v2-20261111.zip && python3 -c "import shutil; shutil.make_archive('../SAT重投包-v2-20261111','zip','.','SAT重投包-v2-20261111')" && ls -la ../SAT重投包-v2-20261111.zip && find "SAT重投包-v2-20261111" -maxdepth 2 | sort | head -40
