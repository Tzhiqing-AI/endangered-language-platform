(function() {
    if (typeof AMap === 'undefined') {
        document.getElementById('languageMap').innerHTML =
            '<div class="d-flex align-items-center justify-content-center h-100 text-muted" style="min-height:420px;">地图组件未加载，请检查网络或配置 AMAP_JS_KEY</div>';
        return;
    }
    // 语言数据 - 由后端 json_script 安全序列化
    var languages = JSON.parse(document.getElementById('map-data').textContent || '[]');

    // 视口接口返回的点是新的对象副本，不能再用 languages.indexOf 取全局下标，
    // 统一走 id -> 全局下标映射，保证点击视口标记仍能高亮右侧列表项
    var languageIndexById = Object.create(null);
    languages.forEach(function(lang, index) {
        if (lang && lang.id !== undefined && lang.id !== null) {
            languageIndexById[lang.id] = index;
        }
    });

    function languageIndexOf(lang) {
        if (lang && lang.id !== undefined && lang.id !== null &&
            languageIndexById[lang.id] !== undefined) {
            return languageIndexById[lang.id];
        }
        return languages.indexOf(lang);
    }

    var dangerColors = {
        '不安全': '#f59e0b',
        '危险': '#f97316',
        '濒危': '#ef4444',
        '垂危': '#dc2626',
        '灭绝': '#991b1b',
        '复苏': '#14b8a6',
        '安全': '#10b981',
        '未分类': '#64748b'
    };

    var dangerWeights = {
        '不安全': 0.2,
        '危险': 0.4,
        '濒危': 0.6,
        '垂危': 0.8,
        '灭绝': 1.0,
        '复苏': 0.1,
        '安全': 0.05,
        '未分类': 0.05
    };

    // 初始化地图：resizeEnable 让 AMap 监听容器尺寸变化；
    // 配合下方的“画布尺寸同步”机制，共同保证地图始终铺满容器
    var MAP_OPTIONS = {
        zoom: 4,
        mapStyle: 'amap://styles/normal',
        center: [105.200213, 36.076518],
        viewMode: '2D',
        resizeEnable: true
    };

    var map = new AMap.Map('languageMap', MAP_OPTIONS);

    // 关闭三维楼块和 POI 图层，减少拖动/缩放时的渲染压力
    map.setFeatures(['bg', 'road']);
    map.on('complete', scheduleMapResize);
    // 聚合模式的视口取数：建图后立即绑定，moveend/zoomend 触发
    bindViewportEvents(map);

    var markers = [];
    var currentFilter = 'all';
    var searchTerm = '';
    var heatmapLayer = null;
    var currentClusterer = null;
    var activeMarker = null;
    var currentViewMode = 'marker';
    var heatmapRenderSeq = 0;
    var clusterRenderSeq = 0;
    var resizeTimer = null;
    var mapSyncFails = 0;

    // 聚合模式的视口点集：默认等于全量，moveend/zoomend 后按 bbox 替换。
    // 仅聚合走视口取数；热力图与普通标记点继续使用全量 languages。
    var clusterViewportLanguages = languages;
    var viewportRequestSeq = 0;
    var viewportDebounceTimer = null;
    // 视口跨度达到该经度值（≈全球视图）时不再请求，直接用全量
    var VIEWPORT_FULL_SPAN = 330;
    var VIEWPORT_DEBOUNCE = 260;

    // 超过该数量时密集点自动聚合，避免低缩放级别下点位堆叠
    var CLUSTER_THRESHOLD = 260;

    // 风险等级颜色与圆点尺寸（风险越高圆点越大）
    var RISK_STYLES = {
        '低风险': { color: '#10b981', size: 9 },
        '中风险': { color: '#f59e0b', size: 11 },
        '高风险': { color: '#f97316', size: 14 },
        '极高风险': { color: '#dc2626', size: 17 }
    };
    var RISK_BAND_KEYS = ['低风险', '中风险', '高风险', '极高风险'];
    var RISK_BAND_RANK = { '低风险': 0, '中风险': 1, '高风险': 2, '极高风险': 3 };

    // ===== 画布尺寸同步（右侧白块的核心修复） =====
    // AMap 1.4 没有公开的 resize()，它靠塞进容器的一个 <object> 监听器感知容器尺寸变化。
    // 一旦该监听器未触发或失效，画布会停留在旧尺寸：右侧露出 AMap 内部白底，
    // 画布之外的语言点也一并被裁掉，看起来就像“点位不对应坐标”。
    // 这里做三层兜底：主动通知监听器 -> 微调容器触发原生 resize -> 连续失败则重建地图。

    function isMapSizeInSync() {
        var el = document.getElementById('languageMap');
        if (!el || !map) return true;
        try {
            var size = map.getSize();
            return Math.abs(size.width - el.clientWidth) < 2 &&
                Math.abs(size.height - el.clientHeight) < 2;
        } catch (e) {
            return true;
        }
    }

    function resizeMap() {
        if (!map) return;
        // map.resize() 是 AMap 2.0 的 API，1.4 下先探测再调用
        if (typeof map.resize === 'function') {
            try { map.resize(); } catch (e) { }
        }
        // AMap 1.4 的 resizeEnable 会在地图容器内插入 <object>，
        // 其 contentWindow 上挂着高德的 resize 监听器，主动派发事件让它重新量尺寸
        var el = document.getElementById('languageMap');
        var obj = el ? el.querySelector('object') : null;
        if (obj && obj.contentDocument && obj.contentDocument.defaultView) {
            try { obj.contentDocument.defaultView.dispatchEvent(new Event('resize')); } catch (e) { }
        }
    }

    // 兜底：把容器宽度微调 2px 再还原，强制 <object> 触发原生 resize 事件
    function nudgeMapContainer() {
        var el = document.getElementById('languageMap');
        if (!el) return;
        try {
            var prev = el.style.width;
            el.style.width = (el.clientWidth - 2) + 'px';
            void el.offsetWidth; // 强制回流
            el.style.width = prev || '';
        } catch (e) { }
    }

    function syncMapSize() {
        if (!map || typeof AMap === 'undefined') return;
        if (isMapSizeInSync()) {
            mapSyncFails = 0;
            return;
        }
        mapSyncFails++;
        resizeMap();
        nudgeMapContainer();
        // 通知/微调后 300ms 复查；连续三次不同步说明监听器已失效，销毁重建地图实例
        setTimeout(function() {
            if (!map || isMapSizeInSync()) {
                mapSyncFails = 0;
                return;
            }
            mapSyncFails++;
            if (mapSyncFails >= 3) {
                mapSyncFails = 0;
                recreateMap();
            }
        }, 300);
    }

    function scheduleMapResize() {
        if (resizeTimer) clearTimeout(resizeTimer);
        resizeTimer = setTimeout(syncMapSize, 150);
    }

    // 最后手段：销毁后在同一容器内重建地图，保留当前视野并重渲染图层
    function recreateMap() {
        var center = null, zoom = null;
        try {
            center = map.getCenter();
            zoom = map.getZoom();
        } catch (e) { }
        try { map.destroy(); } catch (e) { }
        // 清掉所有引用旧地图实例的图层缓存，避免往已销毁的地图上挂载
        heatmapLayer = null;
        currentClusterer = null;
        markers = [];
        map = new AMap.Map('languageMap', MAP_OPTIONS);
        map.setFeatures(['bg', 'road']);
        map.on('complete', scheduleMapResize);
        // 新实例必须重新绑定，否则重建后 moveend/zoomend 不再触发视口取数
        bindViewportEvents(map);
        try {
            if (center && zoom !== null) {
                map.setZoomAndCenter(zoom, [center.lng, center.lat]);
            } else {
                map.setZoomAndCenter(MAP_OPTIONS.zoom, MAP_OPTIONS.center);
            }
        } catch (e) { }
        renderCurrentView();
    }

    window.addEventListener('resize', scheduleMapResize);
    window.addEventListener('orientationchange', scheduleMapResize);
    if (window.ResizeObserver) {
        new ResizeObserver(scheduleMapResize).observe(document.getElementById('languageMap'));
    }
    // 定期体检：字体加载、侧栏晚渲染等导致的容器变化可能逃过事件监听
    setInterval(syncMapSize, 2000);
    setTimeout(syncMapSize, 300);
    setTimeout(syncMapSize, 1200);
    setTimeout(syncMapSize, 4000);

    // 当前筛选 + 搜索后的语言集合
    function getFilteredLanguages() {
        return languages.filter(function(lang) {
            if (currentFilter !== 'all' && lang.endangerment !== currentFilter) return false;
            if (searchTerm && !lang.name.toLowerCase().includes(searchTerm.toLowerCase())) return false;
            return true;
        });
    }

    // 清空当前地图上的所有图层，避免不同视图叠加
    function clearMapLayers() {
        // 使尚未完成的异步插件回调失效
        clusterRenderSeq++;

        if (currentClusterer) {
            currentClusterer.setMap(null);
            currentClusterer = null;
        }

        if (markers.length) {
            map.remove(markers);
            markers = [];
        }

        if (heatmapLayer) {
            heatmapLayer.hide();
        }
    }

    // 构建标记点：用与图例一致的彩色圆点替代默认蓝色图钉
    function buildMarkers(list) {
        var result = [];

        list.forEach(function(lang) {
            var index = languageIndexOf(lang);
            var color = dangerColors[lang.endangerment] || '#64748b';
            var markerOptions = {
                position: new AMap.LngLat(lang.longitude, lang.latitude),
                title: lang.name,
                extData: index,
                content: '<div style="width:14px;height:14px;border-radius:50%;background:' + color +
                    ';border:2px solid #fff;box-shadow:0 1px 4px rgba(15,23,42,.35);"></div>',
                offset: new AMap.Pixel(-9, -9),
                zIndex: 120
            };

            var marker = new AMap.Marker(markerOptions);
            marker.on('click', function() {
                showLanguageDetail(lang);
                highlightListItem(index);
            });
            result.push(marker);
        });

        return result;
    }

    // 聚合气泡：显示聚合数量，避免仅用纯色圆点无法判断密度
    function renderClusterMarker(context) {
        var count = context.count;
        var size = count >= 1000 ? 54 : count >= 100 ? 48 : count >= 10 ? 42 : 34;
        var el = document.createElement('div');
        el.style.cssText = 'width:' + size + 'px;height:' + size + 'px;line-height:' + size + 'px;' +
            'border-radius:50%;background:rgba(15,118,110,.94);border:3px solid #fff;' +
            'color:#fff;font-weight:700;text-align:center;font-size:' + Math.round(size * 0.34) + 'px;' +
            'box-shadow:0 3px 10px rgba(15,23,42,.28);';
        el.textContent = count;
        context.marker.setContent(el);
        if (typeof context.marker.setAnchor === 'function') {
            context.marker.setAnchor('center');
        }
    }

    // 统一聚合入口：异步加载插件期间若视图已切换，则丢弃本次回调
    function addClusterLayer(markerList, gridSize, maxZoom, renderFn) {
        var seq = clusterRenderSeq;
        try {
            map.plugin('AMap.MarkerClusterer', function() {
                if (seq !== clusterRenderSeq) return;
                try {
                    currentClusterer = new AMap.MarkerClusterer(map, markerList, {
                        gridSize: gridSize,
                        maxZoom: maxZoom,
                        averageCenter: true,
                        minClusterSize: 2,
                        renderClusterMarker: renderFn || renderClusterMarker
                    });
                } catch (err) {
                    map.add(markerList);
                }
            });
        } catch (e) {
            map.add(markerList);
        }
    }

    // 聚合模式使用的点集：在视口子集上应用当前筛选/搜索
    function getClusterLanguages() {
        return clusterViewportLanguages.filter(function(lang) {
            if (currentFilter !== 'all' && lang.endangerment !== currentFilter) return false;
            if (searchTerm && !lang.name.toLowerCase().includes(searchTerm.toLowerCase())) return false;
            return true;
        });
    }

    // 标记点模式：数量较多时自动聚合，放大后自动展开为单个彩色点。
    // 只有“进入聚合分支”时才改用视口点集；普通标记点与搜索命中仍用全量，
    // 保证标记模式行为与改动前完全一致。
    function renderMarkers() {
        clearMapLayers();

        var fullFiltered = getFilteredLanguages();
        var useCluster = fullFiltered.length > CLUSTER_THRESHOLD;
        // 搜索词非空时必须用全量，否则命中结果可能不在当前视口内而“消失”
        var filtered = (useCluster && !searchTerm) ? getClusterLanguages() : fullFiltered;
        markers = buildMarkers(filtered);

        if (useCluster) {
            addClusterLayer(markers, 55, 16);
        } else if (markers.length) {
            map.add(markers);
        }

        updateList();
        updateStats();
        scheduleMapResize();
    }

    // 当前视口的 bbox 字符串；视口覆盖全球时返回 null（直接用全量）
    function currentViewportBbox() {
        if (!map || typeof map.getBounds !== 'function') return null;
        var bounds;
        try {
            bounds = map.getBounds();
        } catch (e) {
            return null;
        }
        if (!bounds) return null;
        var sw = bounds.getSouthWest();
        var ne = bounds.getNorthEast();
        var minLng = Number(sw.getLng());
        var maxLng = Number(ne.getLng());
        var minLat = Number(sw.getLat());
        var maxLat = Number(ne.getLat());
        if (!isFinite(minLng) || !isFinite(maxLng) || !isFinite(minLat) || !isFinite(maxLat)) {
            return null;
        }
        if (maxLng - minLng >= VIEWPORT_FULL_SPAN) return null;
        return [minLng, minLat, maxLng, maxLat].map(function(value) {
            return value.toFixed(4);
        }).join(',');
    }

    // 聚合模式下按视口取数：服务端裁剪后返回点集，回来只重建聚合图层
    function requestClusterViewport() {
        if (currentViewMode !== 'marker' || searchTerm) return;
        // 只有进入聚合分支才按视口取数；普通标记点模式仍用全量，避免无谓请求
        if (getFilteredLanguages().length <= CLUSTER_THRESHOLD) return;

        var bbox = currentViewportBbox();
        if (!bbox) {
            // 全球视图：不请求，直接用全量
            viewportRequestSeq++;
            if (clusterViewportLanguages !== languages) {
                clusterViewportLanguages = languages;
                renderMarkers();
            }
            return;
        }

        var seq = ++viewportRequestSeq;
        var url = '/Dashboard/language_map/?mode=cluster&format=json&bbox=' +
            encodeURIComponent(bbox);
        fetch(url, { headers: { 'X-Requested-With': 'XMLHttpRequest' } })
            .then(function(response) {
                return response.ok ? response.json() : null;
            })
            .then(function(data) {
                // 视图已切换或有更新的请求时丢弃本次结果，避免旧视口覆盖新视口
                if (seq !== viewportRequestSeq || currentViewMode !== 'marker') return;
                if (!data || !Array.isArray(data.points)) return;
                clusterViewportLanguages = data.points;
                renderMarkers();
            })
            .catch(function() {
                // 请求失败保留上一次成功的点集，不打断地图交互
            });
    }

    // 绑定视口事件：初始建图与 recreateMap 重建后都要调用，否则重建后视口加载失效
    function bindViewportEvents(targetMap) {
        if (!targetMap || typeof targetMap.on !== 'function') return;
        var schedule = function() {
            if (currentViewMode !== 'marker') return;
            if (viewportDebounceTimer) clearTimeout(viewportDebounceTimer);
            viewportDebounceTimer = setTimeout(requestClusterViewport, VIEWPORT_DEBOUNCE);
        };
        targetMap.on('moveend', schedule);
        targetMap.on('zoomend', schedule);
    }

    // 热力图模式：始终使用当前筛选后的数据
    function renderHeatmap() {
        clearMapLayers();

        var filtered = getFilteredLanguages();
        var heatmapData = filtered.map(function(lang) {
            return {
                lng: lang.longitude,
                lat: lang.latitude,
                weight: dangerWeights[lang.endangerment] || 0.5
            };
        });

        var seq = ++heatmapRenderSeq;

        if (!heatmapLayer) {
            map.plugin('AMap.Heatmap', function() {
                // 如果期间又切换了筛选或视图，丢弃这次异步回调
                if (seq !== heatmapRenderSeq || currentViewMode !== 'heatmap') return;

                heatmapLayer = new AMap.Heatmap(map, {
                    radius: 50,
                    opacity: [0.4, 0.9],
                    gradient: {
                        0.2: '#10b981',
                        0.4: '#f59e0b',
                        0.6: '#f97316',
                        0.8: '#ef4444',
                        1.0: '#dc2626'
                    }
                });
                heatmapLayer.setDataSet({ data: heatmapData, max: 1 });
                heatmapLayer.show();
            });
        } else {
            heatmapLayer.setDataSet({ data: heatmapData, max: 1 });
            heatmapLayer.show();
        }

        updateList();
        updateStats();
    }

    // 风险预测模式：离散风险圆点，而不是与热力图相同的平滑色块
    function buildRiskMarkers(list) {
        var result = [];

        list.forEach(function(lang) {
            var index = languageIndexOf(lang);
            var styleConfig = RISK_STYLES[lang.riskLabel] || { color: '#64748b', size: 10 };
            var s = styleConfig.size;
            var marker = new AMap.Marker({
                position: new AMap.LngLat(lang.longitude, lang.latitude),
                title: lang.name,
                extData: index,
                content: '<div style="width:' + s + 'px;height:' + s + 'px;border-radius:50%;background:' +
                    styleConfig.color + ';border:2px solid #fff;box-shadow:0 1px 4px rgba(15,23,42,.35);"></div>',
                offset: new AMap.Pixel(-s / 2, -s / 2),
                zIndex: 120
            });
            marker.on('click', function() {
                showLanguageDetail(lang);
                highlightListItem(index);
            });
            result.push(marker);
        });

        return result;
    }

    // 风险聚合气泡：按簇内最高风险等级着色，放大后自动还原为分色圆点
    function renderRiskClusterMarker(context) {
        var count = context.count;
        var worstRank = -1;
        var mks = context.markers || (context.cluster && context.cluster.markers) || [];
        for (var i = 0; i < mks.length; i++) {
            var idx = mks[i] && mks[i].getExtData ? mks[i].getExtData() : null;
            if (typeof idx === 'number' && languages[idx]) {
                var rank = RISK_BAND_RANK[languages[idx].riskLabel];
                if (rank !== undefined && rank > worstRank) worstRank = rank;
            }
        }
        var color = worstRank >= 0 ? RISK_STYLES[RISK_BAND_KEYS[worstRank]].color : 'rgba(15,118,110,.94)';
        var size = count >= 1000 ? 54 : count >= 100 ? 48 : count >= 10 ? 42 : 34;
        var el = document.createElement('div');
        el.style.cssText = 'width:' + size + 'px;height:' + size + 'px;line-height:' + size + 'px;' +
            'border-radius:50%;background:' + color + ';border:3px solid #fff;' +
            'color:#fff;font-weight:700;text-align:center;font-size:' + Math.round(size * 0.34) + 'px;' +
            'box-shadow:0 3px 10px rgba(15,23,42,.28);';
        el.textContent = count;
        context.marker.setContent(el);
        if (typeof context.marker.setAnchor === 'function') {
            context.marker.setAnchor('center');
        }
    }

    function updateRiskPanel(list, top) {
        var counts = { '低风险': 0, '中风险': 0, '高风险': 0, '极高风险': 0 };
        list.forEach(function(lang) {
            if (counts.hasOwnProperty(lang.riskLabel)) {
                counts[lang.riskLabel]++;
            }
        });

        document.getElementById('riskCountLow').textContent = counts['低风险'];
        document.getElementById('riskCountMedium').textContent = counts['中风险'];
        document.getElementById('riskCountHigh').textContent = counts['高风险'];
        document.getElementById('riskCountCritical').textContent = counts['极高风险'];

        var listEl = document.getElementById('riskTopList');
        listEl.innerHTML = '';
        top.slice(0, 7).forEach(function(lang) {
            var li = document.createElement('li');
            li.className = 'risk-top-item';
            li.innerHTML = '<span>' + escapeHtml(lang.name) + '</span>' +
                '<span class="risk-top-score">' + Math.round(Number(lang.riskScore)) + '</span>';
            li.onclick = function() {
                focusOnLanguage(lang);
            };
            listEl.appendChild(li);
        });
    }

    function renderRisk() {
        clearMapLayers();

        var filtered = getFilteredLanguages().filter(function(lang) {
            return lang.riskScore !== null && lang.riskScore !== undefined;
        });
        var top = filtered.slice().sort(function(a, b) {
            return Number(b.riskScore) - Number(a.riskScore);
        });

        // 每个语言一个按风险分色/分大小的圆点，坐标即真实经纬度
        var dots = buildRiskMarkers(filtered);
        markers = dots.slice(); // 统一交给 clearMapLayers 清理

        // 点位较多时聚合，密集区显示数量气泡而不是挤成一团
        if (filtered.length > CLUSTER_THRESHOLD) {
            addClusterLayer(dots, 55, 16, renderRiskClusterMarker);
        } else if (dots.length) {
            map.add(dots);
        }

        // 风险最高的语言叠加分数标签，形成“预警点”而不是一片红色热区
        top.slice(0, 10).forEach(function(lang) {
            var index = languages.indexOf(lang);
            var label = new AMap.Marker({
                position: new AMap.LngLat(lang.longitude, lang.latitude),
                title: lang.name + ' · 风险 ' + Math.round(Number(lang.riskScore)),
                content: '<div class="risk-score-label">' + Math.round(Number(lang.riskScore)) + '</div>',
                offset: new AMap.Pixel(-18, -30),
                zIndex: 130
            });
            label.on('click', function() {
                showLanguageDetail(lang);
                highlightListItem(index);
            });
            markers.push(label);
            map.add(label);
        });

        updateRiskPanel(filtered, top);
        updateList();
        updateStats();
        scheduleMapResize();
    }

    function setPanelMode(mode) {
        var legend = document.getElementById('endangermentLegendCard');
        var risk = document.getElementById('riskLegendCard');
        if (mode === 'risk') {
            legend.style.display = 'none';
            risk.style.display = 'block';
        } else {
            legend.style.display = 'block';
            risk.style.display = 'none';
        }
    }

    // 根据当前视图模式刷新地图
    function renderCurrentView() {
        setPanelMode(currentViewMode);
        if (currentViewMode === 'heatmap') {
            renderHeatmap();
        } else if (currentViewMode === 'risk') {
            renderRisk();
        } else {
            renderMarkers();
        }
    }

    // 更新语言列表
    function escapeHtml(value) {
        return String(value == null ? '' : value).replace(/[&<>"']/g, function(ch) {
            return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch];
        });
    }

    function formatSpeakers(value) {
        var n = Number(value);
        return n > 0 ? '约' + n.toLocaleString() + '人' : '数据未知';
    }

    function updateList() {
        var listEl = document.getElementById('langList');
        listEl.innerHTML = '';

        var filteredLangs = languages.filter(function(lang) {
            if (currentFilter !== 'all' && lang.endangerment !== currentFilter) return false;
            if (searchTerm && !lang.name.toLowerCase().includes(searchTerm.toLowerCase())) return false;
            return true;
        });

        document.getElementById('filteredCount').textContent = filteredLangs.length;

        if (filteredLangs.length === 0) {
            listEl.innerHTML = '<li class="empty-state"><div class="empty-icon"><i class="bi bi-search"></i></div>未找到匹配的语言</li>';
            return;
        }

        filteredLangs.forEach(function(lang) {
            var index = languages.indexOf(lang);
            var li = document.createElement('li');
            li.className = 'lang-list-item';
            li.setAttribute('data-index', index);
            var color = dangerColors[lang.endangerment] || '#64748b';
            li.innerHTML = '<div class="lang-dot" style="background:' + color + ';"></div>' +
                '<div class="lang-info"><div class="lang-name">' + escapeHtml(lang.name) + '</div><div class="lang-meta">' + escapeHtml(lang.region) + ' · ' + escapeHtml(formatSpeakers(lang.speakers)) + '</div></div>' +
                '<span class="lang-badge" style="background:' + color + ';">' + escapeHtml(lang.endangerment) + '</span>';
            li.onclick = function() {
                focusOnLanguage(lang);
            };
            listEl.appendChild(li);
        });
    }

    // 聚焦语言
    function focusOnLanguage(lang) {
        map.setCenter([lang.longitude, lang.latitude]);
        map.setZoom(6);
        showLanguageDetail(lang);

        // 高亮列表项
        document.querySelectorAll('.lang-list-item').forEach(function(item) {
            item.classList.remove('active');
        });
        var listItems = document.querySelectorAll('.lang-list-item');
        listItems.forEach(function(item) {
            if (languages[item.getAttribute('data-index')] === lang) {
                item.classList.add('active');
                item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
    }

    // 高亮列表项
    function highlightListItem(index) {
        document.querySelectorAll('.lang-list-item').forEach(function(item) {
            item.classList.remove('active');
        });
        document.querySelectorAll('.lang-list-item[data-index="' + index + '"]').forEach(function(item) {
            item.classList.add('active');
            item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        });
    }

    // 显示语言详情
    function showLanguageDetail(lang) {
        var panel = document.getElementById('langDetailPanel');
        document.getElementById('detailName').textContent = lang.name;
        document.getElementById('detailBadge').textContent = lang.endangerment;
        document.getElementById('detailBadge').style.backgroundColor = dangerColors[lang.endangerment] || '#64748b';
        document.getElementById('detailSpeakers').textContent = formatSpeakers(lang.speakers);
        document.getElementById('detailRegion').textContent = lang.region;
        document.getElementById('detailFamily').textContent = lang.family;
        document.getElementById('detailCoords').textContent = lang.latitude.toFixed(4) + ', ' + lang.longitude.toFixed(4);
        document.getElementById('detailRisk').textContent = (lang.riskScore === null || lang.riskScore === undefined)
            ? '暂无预测'
            : lang.riskLabel + ' · ' + Number(lang.riskScore).toFixed(1);
        panel.classList.add('show');
    }

    window.closeDetail = function() {
        document.getElementById('langDetailPanel').classList.remove('show');
    };

    // 更新统计
    function updateStats() {
        document.getElementById('totalCount').textContent = languages.length;

        var totalSpeakers = languages.reduce(function(sum, lang) {
            var num = parseInt(String(lang.speakers || '').replace(/[^0-9]/g, '')) || 0;
            return sum + num;
        }, 0);

        var countByLevel = {
            'all': languages.length,
            '不安全': 0,
            '危险': 0,
            '濒危': 0,
            '垂危': 0,
            '灭绝': 0,
            '复苏': 0,
            '安全': 0,
            '未分类': 0
        };

        languages.forEach(function(lang) {
            if (countByLevel.hasOwnProperty(lang.endangerment)) {
                countByLevel[lang.endangerment]++;
            }
        });

        document.getElementById('totalSpeakers').textContent = totalSpeakers.toLocaleString();
        document.getElementById('countAll').textContent = countByLevel.all;
        document.getElementById('countUnsafe').textContent = countByLevel['不安全'];
        document.getElementById('countDanger').textContent = countByLevel['危险'];
        document.getElementById('countEndangered').textContent = countByLevel['濒危'];
        document.getElementById('countSerious').textContent = countByLevel['垂危'];
        document.getElementById('countExtinct').textContent = countByLevel['灭绝'];
        document.getElementById('countAwakening').textContent = countByLevel['复苏'];
        document.getElementById('countSafe').textContent = countByLevel['安全'];
        document.getElementById('countUnknown').textContent = countByLevel['未分类'];
    }

    // 切换视图模式
    function switchView(mode, btn) {
        currentViewMode = mode;

        document.querySelectorAll('.toolbar-btn').forEach(function(item) {
            item.classList.remove('active');
        });

        var target = btn || (window.event && window.event.target);
        if (target) {
            target.classList.add('active');
        }

        // 模式切换前显式清空所有图层（聚合/热力/风险圆点），
        // 避免异步插件回调或上一种模式残留图层与新图层叠加
        if (viewportDebounceTimer) clearTimeout(viewportDebounceTimer);
        viewportRequestSeq++;
        clusterViewportLanguages = languages;
        clearMapLayers();

        map.setZoomAndCenter(4, [105.200213, 36.076518]);
        renderCurrentView();
        scheduleMapResize();
    }
    window.switchView = switchView;

    // 搜索功能：保持当前视图模式，并做轻量防抖
    var searchRenderTimer = null;
    document.getElementById('searchInput').addEventListener('input', function(e) {
        searchTerm = e.target.value;
        if (searchRenderTimer) {
            clearTimeout(searchRenderTimer);
        }
        searchRenderTimer = setTimeout(renderCurrentView, 180);
    });

    document.getElementById('searchInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            var firstMatch = languages.find(function(lang) {
                return lang.name.toLowerCase().includes(searchTerm.toLowerCase());
            });
            if (firstMatch) {
                focusOnLanguage(firstMatch);
            }
        }
    });

    // 筛选功能
    document.querySelectorAll('.filter-chip').forEach(function(chip) {
        chip.addEventListener('click', function() {
            document.querySelectorAll('.filter-chip').forEach(function(c) { c.classList.remove('active'); });
            chip.classList.add('active');
            currentFilter = chip.getAttribute('data-filter');
            renderCurrentView();
        });
    });

    // 初始化：默认标记点模式
    renderCurrentView();
    updateStats();
    scheduleMapResize();
    setTimeout(resizeMap, 300);

})();
