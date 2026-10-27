%% SHARP_2D.m
% Spatial Heatmap Analysis of Residence-time Probability (SHARP)
% Written by Nayem H., 2025.
% Based on original SHARP software by github.com/codedbynayem (2025)
% Licensed under AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
%
% AI use disclosure: AI tools were used to assist with developing this
% MATlab script. I am responsible for the accuracy and use of this version
% of SHARP.
%% Clean workspace
clearvars
close all

%% Load sptana
[file, path] = uigetfile('*.mat', 'Select file containing sptana variable');
if isequal(file,0); error('No file selected.'); end
load(fullfile(path,file), 'sptana');

% Find Processed_imaging_data
if endsWith(path, filesep), path = path(1:end-1); end
p = path;
while true
    [pHead, pTail] = fileparts(p);
    if strcmp(pTail, 'Processed_imaging_data')
        processedDir = p; break;
    elseif isempty(pHead) || strcmp(pHead, p)
        error('Processed_imaging_data folder not found in the selected file path.');
    else
        p = pHead;
    end
end

% Directories
masterDir = fullfile(processedDir, 'Master_variables');
saveDir   = fullfile(processedDir, '2D_Heatmap_nucleus');

% Remove old ProbOver40s folder, if present
probDir = fullfile(saveDir, 'ProbOver40s');
if exist(probDir, 'dir')
    try rmdir(probDir, 's'); fprintf('Removed obsolete folder: %s\n', probDir);
    catch ME, warning('Could not remove %s: %s', probDir, ME.message); end
end

% Outputs to keep /create
freqDir    = fullfile(saveDir, 'VisitFrequency');
hotspotDir = fullfile(saveDir, 'Hotspot_Overlays'); % overlay PDFs (2 pages)
if ~exist(freqDir, 'dir');    mkdir(freqDir);    end
if ~exist(hotspotDir, 'dir'); mkdir(hotspotDir); end

% Load imageBins.mat
ibFile = fullfile(masterDir, 'imageBins.mat');
if ~exist(ibFile, 'file'); error('imageBins.mat not found at: %s', ibFile); end
S = load(ibFile);
if isfield(S, 'imageBins')
    imageBins = S.imageBins;
else
    imageBins = [];
    fn = fieldnames(S);
    for k = 1:numel(fn)
        v = S.(fn{k});
        if isstruct(v) && all(isfield(v, {'roiX0','roiY0','width','height'}))
            imageBins = v; break;
        end
    end
    if isempty(imageBins)
        error('No struct with fields roiX0, roiY0, width, height found in imageBins.mat');
    end
end

%% Parameters
pixelsize_nm     = (sptana(1).pixel_size);   % global, same for all cells per user
kde_fwhm_nm      = pixelsize_nm * 3;         % FWHM ~ 3 px
hotspot_percentile      = 97.5;              % percentile in nucleus
hotspot_min_area_um2    = 0.05;              % µm^2 (filter for tiny hotspots)
min_events_per_hotspot  = 3;                 % minimum # events to consider a hotspot
qc_robust_z_thresh      = 11;                % robust outlier cutoff (test based on your own dataset)
pause_entry_thresh  = 40;
productive_elong_thresh = 90;


Nnuc = numel(sptana);

%% Parallel pool
try
    p0 = gcp('nocreate');
    needNew = true;
    if ~isempty(p0)
        % If pool exists but is already suitable and <=5 workers, keep it
        try
            poolType = class(p0); % e.g., 'parallel.ProcessPool' or 'parallel.Pool'
        catch
            poolType = '';
        end
        if contains(poolType,'Process','IgnoreCase',true) && p0.NumWorkers <= 5
            needNew = false;
        else
            delete(p0);
        end
    end
    if needNew
        try
            parpool('Processes', 5);
        catch
            % Fallback to local
            parpool('local', 5);
        end
    end
catch ME
    warning('Could not start a 5-worker process pool: %s. Continuing (parfor will run serially if needed).', ME.message);
end

% Attach this script so workers always see it
p0 = gcp('nocreate');
if ~isempty(p0)
    try, addAttachedFiles(p0, {mfilename('fullpath')}); end %#ok<TRYNC>
end

%% Preallocate per-nucleus outputs (parfor-friendly)
stats_nucleus_c          = cell(Nnuc,1);
stats_total_events       = zeros(Nnuc,1);
stats_area_um2           = nan(Nnuc,1);
init_eff_pct             = nan(Nnuc,1);
elong_eff_pct            = nan(Nnuc,1);
pause_index              = nan(Nnuc,1);

num_hotspots             = zeros(Nnuc,1);
mean_hotspot_area        = nan(Nnuc,1);
avg_events_per_hotspot   = nan(Nnuc,1);

% New inside/outside metrics to be added to CSV
init_in_pct              = nan(Nnuc,1);
init_out_pct             = nan(Nnuc,1);
elong_in_pct             = nan(Nnuc,1);
elong_out_pct            = nan(Nnuc,1);
pause_in_idx             = nan(Nnuc,1);
pause_out_idx            = nan(Nnuc,1);

% Per-nucleus KDE threshold
kde_thr_local            = nan(Nnuc,1);

% Per-nucleus ndwet lists (for optional later use / HotspotRegionMetrics)
ndwet_in_hotspots        = cell(Nnuc,1);
ndwet_outside_hotspot    = cell(Nnuc,1);

% Per-nucleus hotspot pools for QC (gather and concat post-parfor)
all_hs_area_um2_cells    = cell(Nnuc,1);
all_hs_events_cells      = cell(Nnuc,1);

%% Per nucleus processing
parfor i = 1:Nnuc
    fprintf('Processing nucleus %d/%d ...\n', i, Nnuc);

    % Pixel size and units
    if ~isfield(sptana(i),'pixel_size') || isempty(sptana(i).pixel_size)
        error('sptana(%d).pixel_size missing.', i);
    end
    pxSize_nm   = double(sptana(i).pixel_size);
    px_to_um    = pxSize_nm / 1000;

    % KDE kernel (FWHM -> sigma)
    kde_sigma_px = (kde_fwhm_nm/pxSize_nm) / 2.354820045;
    G = local_make_gauss_kernel(kde_sigma_px);

    % Boundary and area
    boundary  = sptana(i).envelop;
    xBoundary = double(boundary(:,1));
    yBoundary = double(boundary(:,2));
    % ROI
    if i > numel(imageBins)
        error('imageBins has fewer entries (%d) than sptana (%d).', numel(imageBins), Nnuc);
    end
    x0 = double(imageBins(i).roiX0);  y0 = double(imageBins(i).roiY0);
    W  = double(imageBins(i).width);  H  = double(imageBins(i).height);

    stats_area_um2(i) = polyarea(xBoundary, yBoundary) * (px_to_um^2);

    % Events (use average XY = cols 8/9 per user)
    X  = double(sptana(i).nucl_summary(:,8));
    Y  = double(sptana(i).nucl_summary(:,9));
    RT = double(sptana(i).ndwet(:));

    stats_total_events(i) = numel(RT);
    n_initiated = sum(RT >= pause_entry_thresh);
    n_paused    = sum(RT >= pause_entry_thresh & RT <= productive_elong_thresh);
    n_elong     = sum(RT >  productive_elong_thresh);
    if stats_total_events(i) > 0
        init_eff_pct(i)  = 100 * n_initiated / stats_total_events(i);
        elong_eff_pct(i) = 100 * n_elong    / stats_total_events(i);
    end
    if n_elong > 0, pause_index(i) = n_paused / n_elong; end

    % Impulse images (within ROI)
    col = round(X - x0);  row = round(Y - y0);
    valid = (col>=1 & col<=W & row>=1 & row<=H & isfinite(RT));
    col = col(valid); row = row(valid);
    RTv = RT(valid);

    counts_img = zeros(H, W, 'double');
    if ~isempty(row)
        lin = sub2ind([H, W], row, col);
        counts_img = accumarray(lin, 1, [H*W, 1]); counts_img = reshape(counts_img, H, W);
    else
        lin = [];
    end

    % KDE 
    freq_img   = conv2(counts_img, G, 'same');

    % Mask and axes in µm
    pixMask = local_mask_from_polygon(W, H, x0, y0, xBoundary, yBoundary);
    xMin = x0; xMax = x0 + W;  yMin = y0; yMax = y0 + H;
    xMin_um = (xMin - x0) * px_to_um;  xMax_um = (xMax - x0) * px_to_um;
    yMin_um = (yMin - y0) * px_to_um;  yMax_um = (yMax - y0) * px_to_um;

    % Visit Frequency plot
    freq_plot = freq_img; freq_plot(~pixMask) = NaN;
    fig = figure('Visible','off'); ax = axes(fig);
    imagesc(ax, [xMin_um xMax_um], [yMin_um yMax_um], freq_plot);
    set(ax,'YDir','normal','Color','k');
    colormap(ax, jet); c = colorbar(ax); ylabel(c, 'KDE(event count)');
    xlabel(ax,'X (\mum)'); ylabel(ax,'Y (\mum)');
    ttl = sprintf('%s (Visit Frequency, KDE FWHM ~%dnm)', local_sanitize_name(sptana,i), round(kde_fwhm_nm));
    title(ax, ttl, 'Interpreter','none');
    axis(ax, 'equal'); xlim(ax, [xMin_um, xMax_um]); ylim(ax, [yMin_um, yMax_um]);
    exportgraphics(fig, fullfile(freqDir, sprintf('%s.pdf', local_sanitize_name(sptana,i))), 'ContentType','vector');
    close(fig);

    % Hotspot detection
    vals_in = freq_img(pixMask);
    if isempty(vals_in) || all(~isfinite(vals_in))
        thr = Inf;
    else
        thr = prctile(vals_in, hotspot_percentile);
    end
    kde_thr_local(i) = thr;
    if ~isfinite(kde_thr_local(i)), kde_thr_local(i) = NaN; end

    hs_mask = pixMask & (freq_img >= thr);

    % Remove tiny specks by area
    min_area_px = ceil(hotspot_min_area_um2 / (px_to_um^2));
    if min_area_px > 1
        hs_mask = bwareaopen(hs_mask, min_area_px, 8);
    end

    % Initial components on the KDE thresholded mask
    CC0 = bwconncomp(hs_mask, 8);

    %  Count events per hotspot (for thresholding by event count) 
    ev_counts_per_hs0 = [];
    if CC0.NumObjects > 0
        L0 = labelmatrix(CC0);
        if ~isempty(lin)
            ev_labels0 = L0(lin);
            ev_counts_per_hs0 = accumarray(ev_labels0(ev_labels0>0), 1, [CC0.NumObjects,1], @sum, 0);
        else
            ev_counts_per_hs0 = zeros(CC0.NumObjects,1);
        end
    else
        ev_counts_per_hs0 = zeros(0,1);
    end

    %  Keep only components with ≥ min_events_per_hotspot events 
    keep = ev_counts_per_hs0 >= min_events_per_hotspot;
    hs_mask_keep = false(H,W);
    if any(keep)
        for kk = find(keep)'
            hs_mask_keep(CC0.PixelIdxList{kk}) = true;
        end
    end
    CC = bwconncomp(hs_mask_keep, 8);  % filtered components
    num_hotspots(i) = CC.NumObjects;

    % Recompute label image for kept components (for plotting/table)
    L = labelmatrix(CC);
    if ~isempty(lin) && CC.NumObjects > 0
        ev_labels = L(lin);                 % label per event (0 = not in hotspot)
        in_hs     = ev_labels > 0;
    else
        in_hs     = false(size(RTv));
    end

    % Save ndwet inside/outside
    ndwet_in_hotspots{i}      = RTv(in_hs);
    ndwet_outside_hotspot{i}  = RTv(~in_hs);

    % - Collect per-hotspot stats after filtering -
    areas_um2 = []; ev_counts_per_hs = [];
    if CC.NumObjects > 0
        Sprops = regionprops(CC, 'Area','Centroid');
        areas_px = [Sprops.Area]';
        areas_um2 = areas_px * (px_to_um^2);

        % counts per kept hotspot
        if ~isempty(lin)
            ev_labels = L(lin);
            ev_counts_per_hs = accumarray(ev_labels(ev_labels>0), 1, [CC.NumObjects,1], @sum, 0);
            avg_events_per_hotspot(i) = mean(ev_counts_per_hs);
        else
            ev_counts_per_hs = zeros(CC.NumObjects,1);
            avg_events_per_hotspot(i) = NaN;
        end
        mean_hotspot_area(i)   = mean(areas_um2);

        % - Append to global pools for QC (store per-nucleus, concat later) -
        all_hs_area_um2_cells{i} = areas_um2(:);
        all_hs_events_cells{i}   = ev_counts_per_hs(:);
    else
        mean_hotspot_area(i)   = NaN;
        avg_events_per_hotspot(i) = NaN;
        all_hs_area_um2_cells{i} = [];
        all_hs_events_cells{i}   = [];
    end

    % Map boundaries to kept labels (for numbered outlines)
    B = bwboundaries(hs_mask_keep, 'noholes');
    boundaries_by_label = cell(CC.NumObjects,1);
    for kk = 1:numel(B)
        b = B{kk}; % [row col]
        r0 = max(1, min(H, round(b(1,1))));
        c0 = max(1, min(W, round(b(1,2))));
        lid = L(r0, c0);
        if lid > 0
            boundaries_by_label{lid} = b;
        end
    end

    % Hotspot overlay PDF (Page 1)
    overlayFile = fullfile(hotspotDir, sprintf('%s.pdf', local_sanitize_name(sptana,i)));
    figO = figure('Visible','off'); axO = axes(figO);
    imagesc(axO, [xMin_um xMax_um], [yMin_um yMax_um], freq_plot);
    set(axO,'YDir','normal','Color','k'); axis(axO,'equal'); hold(axO,'on');
    colormap(axO, local_hot_capped());
    cO = colorbar(axO); ylabel(cO, 'KDE(event count)');
    clim = caxis(axO);
    caxis(axO, [clim(1), clim(1) + 0.95*(clim(2)-clim(1))]);
    xlabel(axO,'X (\mum)'); ylabel(axO,'Y (\mum)');
    xb_um = (xBoundary - x0) * px_to_um;  yb_um = (yBoundary - y0) * px_to_um;
    plot(axO, xb_um, yb_um, 'w-', 'LineWidth', 1.2);

    label_offset_px = 5;
    if CC.NumObjects > 0
        Sprops2 = regionprops(CC, 'Area','Centroid');
        for h = 1:CC.NumObjects
            b = boundaries_by_label{h}; if isempty(b), continue; end
            x_um = (b(:,2) - 0.5) * px_to_um;
            y_um = (b(:,1) - 0.5) * px_to_um;
            patch(axO, x_um, y_um, [1 0 0], 'FaceAlpha', 0, ...
                  'EdgeColor', [0 1 1], 'LineWidth', 1.3);
            Cpx = Sprops2(h).Centroid;  dx = b(:,2)-Cpx(1); dy = b(:,1)-Cpx(2);
            [~, idxMax] = max(dx.^2 + dy.^2);
            bp  = [b(idxMax,2), b(idxMax,1)];
            nv  = [dx(idxMax), dy(idxMax)]; nv = nv / max(eps, norm(nv));
            p_out = bp + label_offset_px * nv;
            tx = (p_out(1) - 0.5) * px_to_um;  ty = (p_out(2) - 0.5) * px_to_um;
            text(axO, tx, ty, sprintf('%d',h), 'Color','w','FontSize',8,'FontWeight','bold', ...
                'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'BackgroundColor','none', 'Clipping','on');
        end
    end
    title(axO, sprintf('%s | Hotspots >= %.1f%%ile | min area %.3f \\mum^2 | N=%d', ...
        local_sanitize_name(sptana,i), hotspot_percentile, hotspot_min_area_um2, CC.NumObjects), ...
        'Interpreter','none');
    xlim(axO, [xMin_um, xMax_um]); ylim(axO, [yMin_um, yMax_um]);
    exportgraphics(figO, overlayFile, 'ContentType','vector');  % first page
    close(figO);

    % Page 2: text table (+ XLSX per nucleus)
    areas_um2_tbl = []; ev_counts_tbl = [];
    if CC.NumObjects > 0
        Sprops3 = regionprops(CC, 'Area');
        areas_um2_tbl = [Sprops3.Area]' * (px_to_um^2);
        if ~isempty(lin)
            ev_labels = L(lin);
            ev_counts_tbl = accumarray(ev_labels(ev_labels>0), 1, [CC.NumObjects,1], @sum, 0);
        else
            ev_counts_tbl = zeros(CC.NumObjects,1);
        end
    end
    hs_ids = (1:numel(areas_um2_tbl)).';  % matches the cyan labels / page-2 IDs
    T_hs = table(hs_ids, areas_um2_tbl, ev_counts_tbl, ...
        'VariableNames', {'Hotspot_ID','Area_um2','Events'});
    xlsxFile = fullfile(hotspotDir, sprintf('%s_hotspots.xlsx', local_sanitize_name(sptana,i)));
    writetable(T_hs, xlsxFile);
    figT = figure('Visible','off','Color','w'); axT = axes(figT,'Visible','off'); axis(axT,'off');
    lines = {};
    lines{end+1} = sprintf('%s — Hotspot summary', local_sanitize_name(sptana,i));
    lines{end+1} = '';
    header = sprintf('%3s  %10s  %7s', 'ID','Area_um2','Events');
    lines{end+1} = header; lines{end+1} = repmat('-',1,length(header));
    if ~isempty(areas_um2_tbl)
        if isempty(ev_counts_tbl), ev_counts_tbl = zeros(numel(areas_um2_tbl),1); end
        for h = 1:numel(areas_um2_tbl)
            lines{end+1} = sprintf('%3d  %10.3f  %7d', h, areas_um2_tbl(h), ev_counts_tbl(h));
        end
    else
        lines{end+1} = '(No hotspots detected at current settings.)';
    end
    text(axT, 0.02, 0.98, strjoin(lines, newline), 'Units','normalized', ...
        'VerticalAlignment','top', 'FontName','Courier', 'FontSize',10, ...
        'Interpreter','none', 'Color','k');
    exportgraphics(figT, overlayFile, 'Append', true); % second page
    close(figT);

    % Save nucleus name
    stats_nucleus_c{i} = string(local_sanitize_name(sptana,i));

    %  Inside vs Outside metrics for this nucleus 
    rin = ndwet_in_hotspots{i};
    rout = ndwet_outside_hotspot{i};

    % Inside hotspots
    Ni   = numel(rin);
    if Ni > 0
        nInit_i = sum(rin >= pause_entry_thresh);
        nElon_i = sum(rin >  productive_elong_thresh);
        nPause_i= sum(rin >= pause_entry_thresh & rin <= productive_elong_thresh);
        init_in_pct(i)  = 100 * nInit_i / Ni;
        elong_in_pct(i) = 100 * nElon_i / Ni;
        if nElon_i > 0
            pause_in_idx(i) = nPause_i / nElon_i;
        else
            pause_in_idx(i) = NaN;
        end
    end

    % Outside hotspots
    No   = numel(rout);
    if No > 0
        nInit_o = sum(rout >= pause_entry_thresh);
        nElon_o = sum(rout >  productive_elong_thresh);
        nPause_o= sum(rout >= pause_entry_thresh & rout <= productive_elong_thresh);
        init_out_pct(i)  = 100 * nInit_o / No;
        elong_out_pct(i) = 100 * nElon_o / No;
        if nElon_o > 0
            pause_out_idx(i) = nPause_o / nElon_o;
        else
            pause_out_idx(i) = NaN;
        end
    end
end

%%  QC: mark nuclei FAIL if any hotspot is a robust outlier on area or events 
all_hs_area_um2 = cat(1, all_hs_area_um2_cells{:});
all_hs_events   = cat(1, all_hs_events_cells{:});
all_hs_parent_idx = [];
if ~isempty(all_hs_area_um2)
    % Build parent index vector (repeat nucleus index per-hotspot)
    counts_per_nuc = cellfun(@numel, all_hs_area_um2_cells);
    all_hs_parent_idx = arrayfun(@(idx,n) repmat(idx,n,1), (1:Nnuc)', counts_per_nuc(:), 'UniformOutput', false);
    all_hs_parent_idx = cat(1, all_hs_parent_idx{:});

    zA = local_robust_zscore(all_hs_area_um2);
    zE = local_robust_zscore(all_hs_events);
    outlier_idx = (zA > qc_robust_z_thresh) | (zE > qc_robust_z_thresh);
    bad_nuclei = unique(all_hs_parent_idx(outlier_idx));
    qc_flag = repmat("PASS", Nnuc, 1);
    qc_flag(bad_nuclei) = "FAIL";
else
    qc_flag = repmat("PASS", Nnuc, 1);  % no hotspots at all -> nothing to fail on
end

%% Normalize hotspots to nuclear area (density)
hotspots_per_um2 = num_hotspots ./ stats_area_um2;
hotspots_per_um2(~isfinite(hotspots_per_um2)) = NaN;  % safety

%% Assemble per-nucleus table (CSV)
stats_nucleus = string(stats_nucleus_c);
T = table(stats_nucleus, stats_total_events, stats_area_um2, ...
    elong_eff_pct, pause_index, ...
    num_hotspots, hotspots_per_um2, mean_hotspot_area, ...
    avg_events_per_hotspot, ...
    elong_in_pct, elong_out_pct, pause_in_idx, pause_out_idx, qc_flag, ...
    'VariableNames', {'Nucleus','Total_events','Nuclear_area_um2', ...
    'Pct_productive_elongation','Pause_index', ...
    'Num_hotspots','Hotspots_per_um2','Mean_hotspot_area_um2', ...
    'Avg_events_per_hotspot', ...
    'Pct_productive_elongation_inside_hotspots','Pct_productive_elongation_outside_hotspots', ...
    'Pause_index_inside_hotspots','Pause_index_outside_hotspots','Nuclei_hotspot_QC'});

% Human-readable descriptive header row
descriptiveHeaders = { ...
    'Nucleus', ...
    'Nuclear events', ...
    'Nuclear area (square microns)', ...
    'Percent Pol II reaching productive elongation', ...
    'Pause index', ...
    'Total number of hotspots (KDE-defined)', ...
    'Hotspots per square micron (normalized)', ...
    'Mean hotspot area (square microns)', ...
    'Average number of events per hotspot', ...
    '% reaching productive elongation inside hotspots (%)', ...
    '% reaching productive elongation outside hotspots (%)', ...
    'Pause index inside hotspots', ...
    'Pause index outside hotspots', ...
    'Nuclei hotspot QC (PASS/FAIL)'};

csvFile = fullfile(saveDir, 'hotspot_stats.csv');
writetable(T, csvFile);
fid = fopen(csvFile,'r+'); fgetl(fid); rest = fread(fid, '*char')';
frewind(fid); fprintf(fid, '%s\n', strjoin(descriptiveHeaders, ','));
fwrite(fid, rest); fclose(fid);

%% Local hotspot threshold list
thr_mean_all = mean(kde_thr_local, 'omitnan');
thr_std_all  = std(kde_thr_local,  'omitnan');

Tthr = table( ...
    stats_nucleus, ...
    kde_thr_local, ...
    repmat(thr_mean_all, Nnuc, 1), ...
    repmat(thr_std_all,  Nnuc, 1), ...
    'VariableNames', {'Nucleus','KDE_threshold_local', ...
                      'KDE_threshold_mean_allcells','KDE_threshold_std_allcells'});

writetable(Tthr, fullfile(saveDir, 'kde_thresholds.csv'));

%% Correlation CSV (PASS nuclei only): Elongation vs {Mean hotspot area, Avg events per hotspot}
pass_mask = strcmp(T.Nuclei_hotspot_QC, "PASS");
metricsA = {'Mean_hotspot_area_um2','Avg_events_per_hotspot'};
A_full   = [T.Mean_hotspot_area_um2, T.Avg_events_per_hotspot];
B_full   = T.Elongation_efficiency;

A = A_full(pass_mask, :);
B = B_full(pass_mask);

% Drop rows with any NaN across predictors or outcome
valid = isfinite(B) & all(isfinite(A),2);
A = A(valid,:); B = B(valid);

rowsA = {}; rvals = []; pvals = []; nvals = [];
for a = 1:size(A,2)
    [r,p,n] = local_corr(A(:,a), B);
    rowsA{end+1,1} = metricsA{a}; %#ok<AGROW>
    rvals(end+1,1) = r;            %#ok<AGROW>
    pvals(end+1,1) = p;            %#ok<AGROW>
    nvals(end+1,1) = n;            %#ok<AGROW>
end

TC = table(rowsA, repmat({'Percent_reaching_productive_elongation'}, numel(rowsA), 1), rvals, pvals, nvals, ...
    'VariableNames', {'Hotspot_metric','Biological_metric','Pearson_r','p_value','N_pairs'});

% Benjamini-Hochberg FDR
pvec  = TC.p_value(:);
maskp = isfinite(pvec);
qvec  = nan(size(pvec));
ps = pvec(maskp);
if ~isempty(ps)
    [ps_sorted, order] = sort(ps, 'ascend');   % permutation from original -> sorted
    m = numel(ps_sorted);
    q_sorted = ps_sorted .* m ./ (1:m)'; 
    q_sorted = min(q_sorted, 1);
    for k = m-1:-1:1
        q_sorted(k) = min(q_sorted(k), q_sorted(k+1));  % enforce monotonicity
    end
    invorder = zeros(1, m);
    invorder(order) = 1:m;
    qvec(maskp) = q_sorted(invorder);
end
TC.q_value = qvec;
TC.sig_q10 = qvec <= 0.10;

writetable(TC, fullfile(saveDir, 'hotspot_correlations.csv'));

%% Inside vs. Outside hotspots metrics by nucleus/condition (returned to workspace)
cond_col  = strings(Nnuc,1);
for i = 1:Nnuc
    if isfield(sptana(i),'condition') && ~isempty(sptana(i).condition)
        try
            cond_col(i) = string(sptana(i).condition{1});
        catch
            cond_col(i) = stats_nucleus(i);
        end
    else
        cond_col(i) = stats_nucleus(i);
    end
end

HotspotRegionMetrics = table( ...
    cond_col, pause_in_idx, pause_out_idx, elong_in_pct, elong_out_pct, ...
    'VariableNames', { ...
        'Condition', ...
        'PauseIndex_Inside', ...
        'PauseIndex_Outside', ...
        'ProductiveElongPct_Inside', ...
        'ProductiveElongPct_Outside' ...
    });

clear p0 poolType needNew counts_per_nuc i a ps ps_sorted order invorder ...
      pass_mask A_full B_full A B valid rowsA rvals pvals nvals maskp qvec ...
      x0 y0 W H boundary xBoundary yBoundary xMin xMax yMin yMax xMin_um xMax_um yMin_um yMax_um ...
      X Y RT n_initiated n_paused n_elong col row valid lin counts_img freq_img pixMask freq_plot ...
      thr hs_mask min_area_px CC0 L0 ev_labels0 ev_counts_per_hs0 keep hs_mask_keep CC L ev_labels in_hs ...
      Sprops Sprops2 Sprops3 areas_px areas_um2 ev_counts_per_hs areas_um2_tbl ev_counts_tbl hs_ids ...
      fig ax c ttl figO axO cO clim xb_um yb_um label_offset_px B boundaries_by_label bp nv p_out tx ty figT axT header lines ...
      overlayFile xlsxFile thr_mean_all thr_std_all

fprintf('Analysis complete.\n');
clearvars -except sptana HotspotRegionMetrics
%%  Local functions 
function K = local_make_gauss_kernel(sigma_px)
if sigma_px <= 0 || ~isfinite(sigma_px), K = 1; return; end
rad = max(1, ceil(6*sigma_px));
[xg, yg] = meshgrid(-rad:rad, -rad:rad);
K = exp( -(xg.^2 + yg.^2) / (2*sigma_px^2) ); K = K / sum(K(:));
end

function M = local_mask_from_polygon(W, H, x0, y0, xB, yB)
[Xc, Yc] = meshgrid((x0+0.5):(x0+W-0.5), (y0+0.5):(y0+H-0.5));
M = inpolygon(Xc, Yc, xB, yB);
end

function name = local_sanitize_name(sptana,i)
if isfield(sptana(i),'condition') && ~isempty(sptana(i).condition)
    try, name = regexprep(sptana(i).condition{1}, '[^\w\-\.\(\) ]', '_');
    catch, name = sprintf('Nucleus_%d', i); end
else, name = sprintf('Nucleus_%d', i); end
end

function [r,p,n] = local_corr(x,y)
m = isfinite(x) & isfinite(y); n = sum(m);
if n >= 3, [r,p] = corr(x(m), y(m), 'Type','Pearson'); else, r = NaN; p = NaN; end
end

function cm = local_hot_capped()
    cm = hot(256);
    cm = cm(1:235, :);                    % keep ~92% (no near-white)
    cm = interp1(linspace(0,1,size(cm,1)), cm, linspace(0,1,256));
end

function z = local_robust_zscore(x)
% Returns robust |z| using median/MAD; NaN-safe; vectorized
x = x(:);
z = nan(size(x));
ok = isfinite(x);
if nnz(ok) < 3
    z(ok) = 0; return
end
xo = x(ok);
med = median(xo);
mad0 = median(abs(xo - med));
if mad0 > 0
    sigma = 1.4826 * mad0;
else
    iq = iqr(xo);
    if iq > 0
        sigma = iq / 1.349;
    else
        sigma = std(xo);
        if sigma == 0
            z(ok) = 0; return
        end
    end
end
z(ok) = abs(x(ok) - med) / sigma;
end
