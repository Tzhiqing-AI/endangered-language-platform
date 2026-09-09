"""
管理员专家认证管理视图
"""
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.utils import timezone

from User.models import User, ExpertApplication
from Dashboard.common import get_user_context


@require_http_methods(["GET"])
def expert_management(request):
    """专家认证管理页面"""
    user, name, _ = get_user_context(request)

    if not user:
        return redirect('/User/login/?next=/User/expert/management/')

    if user.admin != 1:
        messages.error(request, '您没有权限访问此页面')
        return redirect('/Dashboard/language/')

    # 获取各状态的申请
    pending_applications = ExpertApplication.objects.filter(status='pending').order_by('-created_at')
    approved_applications = ExpertApplication.objects.filter(status='approved').order_by('-reviewed_at')[:20]
    rejected_applications = ExpertApplication.objects.filter(status='rejected').order_by('-reviewed_at')[:20]

    # 统计数据
    stats = {
        'pending': pending_applications.count(),
        'approved': ExpertApplication.objects.filter(status='approved').count(),
        'rejected': ExpertApplication.objects.filter(status='rejected').count(),
    }

    context = {
        'name': name,
        'username': user.username,
        'pending_applications': pending_applications,
        'approved_applications': approved_applications,
        'rejected_applications': rejected_applications,
        'stats': stats,
    }
    return render(request, 'user/expert_management.html', context)


@require_http_methods(["GET", "POST"])
def expert_application_review(request, pk):
    """审核专家申请"""
    user, name, _ = get_user_context(request)

    if not user:
        return redirect('/User/login/?next=/User/expert/management/')

    if user.admin != 1:
        return JsonResponse({'status': 'error', 'message': '无权审核'}, status=403)

    application = get_object_or_404(ExpertApplication, pk=pk)

    if request.method == 'GET':
        context = {
            'name': name,
            'username': user.username,
            'application': application,
        }
        return render(request, 'user/expert_application_review.html', context)

    # POST - 处理审核
    action = request.POST.get('action')
    comment = request.POST.get('comment', '')

    if action not in ['approve', 'reject']:
        messages.error(request, '无效的操作')
        return redirect('/User/expert/management/')

    from django.db import transaction
    with transaction.atomic():
        application.status = 'approved' if action == 'approve' else 'rejected'
        application.reviewer = user
        application.review_comment = comment
        application.reviewed_at = timezone.now()
        application.save()

        if action == 'approve':
            application.applicant.expert = 1
            application.applicant.expert_verified_at = timezone.now()
            application.applicant.save()

            from Community.models import Message
            Message.objects.create(
                recipient=application.applicant,
                msg_type='system',
                content='您的专家申请已通过！您现在可以在社区中发布权威认证的内容了。'
            )
        else:
            from Community.models import Message
            Message.objects.create(
                recipient=application.applicant,
                msg_type='system',
                content=f'您的专家申请未通过。审核备注：{comment or "无"}'
            )

    action_text = '通过' if action == 'approve' else '拒绝'
    messages.success(request, f'申请已{action_text}')

    return redirect('/User/expert/management/')