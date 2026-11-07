var chartData = {};
try {
    chartData = JSON.parse(document.getElementById('chart-data').textContent || '{}');
} catch (e) { chartData = {}; }

    document.getElementById('downloadCsvErrors')?.addEventListener('click', function() {
        var source = document.getElementById('csv-import-errors');
        if (!source) return;
        var errors = JSON.parse(source.textContent || '[]');
        var content = '\ufeff错误信息\r\n' + errors.map(function(e) {
            return '"' + String(e).replace(/"/g, '""') + '"';
        }).join('\r\n');
        var blob = new Blob([content], {type: 'text/csv;charset=utf-8'});
        var link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'csv_import_errors.csv';
        link.click();
        URL.revokeObjectURL(link.href);
    });

    // 清理模态框残留状态
    function cleanupModal() {
        // 移除所有残留的 backdrop
        var backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(function(b) { b.remove(); });
        // 移除 body 上的 modal-open class
        document.body.classList.remove('modal-open');
        document.body.style.removeProperty('padding-right');
        document.body.style.removeProperty('overflow');
    }

    // 模态框显示函数
    function showModal(id, rowNumber) {
        // 先清理之前的残留状态
        cleanupModal();

        var modal = document.getElementById(id);
        if (modal && typeof bootstrap !== 'undefined') {
            // 如果是删除模态框，需要处理行号到真实ID的转换
            if (id === 'deleteModal' && rowNumber) {
                var row = document.querySelector('tr[data-language-id]');
                // 找到对应行号的那一行
                var rows = document.querySelectorAll('table tbody tr[data-language-id]');
                var targetRow = rows[rowNumber - 1];
                if (targetRow) {
                    var realId = targetRow.getAttribute('data-language-id');
                    document.getElementById('deleteRowNumber').value = rowNumber;
                    document.getElementById('deleteLanguageId').value = realId;
                }
            }

            var bsModal = new bootstrap.Modal(modal);
            bsModal.show();

            // 监听隐藏事件，确保清理
            modal.addEventListener('hidden.bs.modal', function() {
                cleanupModal();
            }, { once: true });
        }
    }

    // 设置删除模态框的行号
    function setDeleteRow(rowNumber) {
        document.getElementById('deleteConfirmNumber').textContent = rowNumber;
        // 找到对应行的真实language_id
        var rows = document.querySelectorAll('table tbody tr[data-language-id]');
        var targetRow = rows[rowNumber - 1];
        if (targetRow) {
            var realId = targetRow.getAttribute('data-language-id');
            document.getElementById('deleteLanguageId').value = realId;
        }
    }

    // 打开删除模态框
    function openDeleteModal(rowNumber) {
        cleanupModal();
        setDeleteRow(rowNumber);
        var modal = document.getElementById('deleteModal');
        if (modal && typeof bootstrap !== 'undefined') {
            var bsModal = new bootstrap.Modal(modal);
            bsModal.show();
            modal.addEventListener('hidden.bs.modal', function() {
                cleanupModal();
            }, { once: true });
        }
    }

    // 打开修改模态框
    function openUpdateModal(rowNumber) {
        cleanupModal();

        // 找到对应行的真实language_id
        var rows = document.querySelectorAll('table tbody tr[data-language-id]');
        var targetRow = rows[rowNumber - 1];
        if (!targetRow) {
            alert('未找到对应的语言数据');
            return;
        }

        var realId = targetRow.getAttribute('data-language-id');
        document.getElementById('updateConfirmNumber').textContent = rowNumber;
        document.getElementById('updateLanguageId').value = realId;

        // 从行内 data 属性填充表单，避免从展示文本反推导致数据被改写
        var data = targetRow.dataset;
        document.getElementById('updateLanguageName').value = data.languageName || '';
        document.getElementById('updateLegacyLevel').value = data.legacyLevel || '';
        document.getElementById('updateLevelOfEndangerment').value = data.elcatLevel || '';
        document.getElementById('updateNumberOfSpeakers').value = data.speakers || '';
        document.getElementById('updateSpeakerTrends').value = data.trend || '';
        document.getElementById('updateIntergenerationalTransmission').value = data.transmission || '';
        document.getElementById('updateLanguageFamily').value = data.family || '';
        document.getElementById('updateCountry').value = data.country || '';
        document.getElementById('updateAdditionalNotes').value = data.notes || '';

        var modal = document.getElementById('updateModal');
        if (modal && typeof bootstrap !== 'undefined') {
            var bsModal = new bootstrap.Modal(modal);
            bsModal.show();
            modal.addEventListener('hidden.bs.modal', function() {
                cleanupModal();
            }, { once: true });
        }
    }

    // 页面加载完成后初始化
    document.addEventListener('DOMContentLoaded', function() {
        // 为所有模态框添加隐藏事件监听
        var modals = document.querySelectorAll('.modal');
        modals.forEach(function(modal) {
            modal.addEventListener('hidden.bs.modal', function() {
                cleanupModal();
            });
        });

        // 删除模态框打开时，自动填入hidden字段
        var deleteModal = document.getElementById('deleteModal');
        if (deleteModal) {
            deleteModal.addEventListener('show.bs.modal', function(event) {
                var button = event.relatedTarget;
                if (button) {
                    var rowNumber = button.getAttribute('data-row-number');
                    if (rowNumber) {
                        setDeleteRow(parseInt(rowNumber));
                    }
                }
            });
        }
    });

    // ECharts 图表初始化
    window.addEventListener('load', function() {
        var chartDom = document.getElementById('chart-container');
        if (chartDom && typeof echarts !== 'undefined') {
            var myChart = echarts.init(chartDom);
            myChart.setOption({
                title: {
                    text: '濒危语言统计分布',
                    left: 'center',
                    textStyle: { fontSize: 18, fontWeight: 'normal', color: '#1e293b' }
                },
                tooltip: { trigger: 'item', formatter: '{a} <br/>{b}: {c} ({d}%)' },
                legend: {
                    orient: 'vertical',
                    right: '5%',
                    top: 'center',
                    data: ['不安全', '危险', '濒危', '垂危', '灭绝', '复苏', '未分类']
                },
                series: [{
                    name: '濒危语言',
                    type: 'pie',
                    radius: ['40%', '70%'],
                    center: ['40%', '50%'],
                    itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
                    label: { show: true, position: 'outside', formatter: '{b}\n{c} ({d}%)', fontSize: 12 },
                    data: [
                        { value: chartData.unsafe || 0, name: '不安全', itemStyle: { color: '#f59e0b' } },
                        { value: chartData.danger || 0, name: '危险', itemStyle: { color: '#f97316' } },
                        { value: chartData.endangered || 0, name: '濒危', itemStyle: { color: '#ef4444' } },
                        { value: chartData.seriously_endangered || 0, name: '垂危', itemStyle: { color: '#dc2626' } },
                        { value: chartData.extinct || 0, name: '灭绝', itemStyle: { color: '#991b1b' } },
                        { value: chartData.legacy_awakening || 0, name: '复苏', itemStyle: { color: '#14b8a6' } },
                        { value: chartData.unknown || 0, name: '未分类', itemStyle: { color: '#64748b' } }
                    ]
                }]
            });
            window.addEventListener('resize', function() { myChart.resize(); });
        }
    });
