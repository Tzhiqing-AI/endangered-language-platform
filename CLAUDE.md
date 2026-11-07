# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Django 6.0 project — "濒危语言保护平台" (Endangered Language Protection Network). Provides language catalog, community forum, REST API, and AI-powered features (translation, chat).

## Commands

### Development
```bash
python manage.py runserver          # Start dev server (default port 8000)
python manage.py makemigrations     # Create migrations for model changes
python manage.py migrate            # Apply pending migrations
python manage.py check              # Run Django system checks
python manage.py check --deploy     # Run deploy security checks
```

### Testing
```bash
pytest --ds=language.settings_test -v   # Run all tests
pytest --ds=language.settings_test Dashboard/tests.py -v  # Run Dashboard tests only
pytest --ds=language.settings_test rest_api/tests.py -v   # Run API tests only
pytest --ds=language.settings_test -k "test_login" -v     # Run by keyword
```
Uses `language.settings_test` (SQLite in-memory DB, MD5 password hasher for speed).

### Migrations
```bash
python manage.py makemigrations Dashboard --name descriptive_name
python manage.py migrate
python manage.py showmigrations
```

## Architecture

### App Layout
```
language/
├── language/          # Project config (settings, urls, wsgi/asgi)
├── Dashboard/         # Core: Language model, AI views, admin dashboard
├── User/             # User model, auth, profile, expert certification
├── Community/        # Forum: posts, replies, likes, follows, messages
├── rest_api/         # DRF REST API (serializers, views, permissions)
├── templates/        # Global templates
│   ├── community/    # Community app templates
│   └── rest_api/     # API templates
├── static/           # Global static files
├── init_community_data.py  # Seed data script
└── manage.py
```

### URL Routing
- `language/urls.py` → includes all app URLconfs
  - `Dashboard/` → `Dashboard.urls`
  - `User/` → `User.urls`
  - `Community/` → `Community.urls`
  - `api/` → `rest_api.urls`

### Key Models
| Model | App | Table |
|-------|-----|-------|
| `User` | User | `user` |
| `Language` | Dashboard | `language` |
| `LanguageLocation` | Dashboard | `language_location` |
| `Post` | Community | `post` |
| `Reply` | Community | `reply` |
| `Like` | Community | `like` |
| `Collect` | Community | `collect` |
| `Follow` | Community | `follow` |
| `Message` | Community | `message` |
| `Report` | Community | `report` |
| `Category` | Community | `category` |
| `Tag` | Community | `tag` |
| `ContentModeration` | Community | `content_moderation` |
| `ExpertApplication` | User | `expert_application` |
| `UserProfile` | User | `user_profile` |
| `TranslationMemory` | Dashboard | `translation_memory` |
| `Glossary` | Dashboard | `glossary` |

### Settings Highlights
- Database: MySQL (production) / SQLite (test) — configurable via `.env`
- Timezone: `Asia/Shanghai`, Language: `zh-hans`
- REST Framework: Session auth, `PageNumberPagination` (PAGE_SIZE=20)
- API docs: coreapi schema
- Static: `STATIC_ROOT = BASE_DIR/staticfiles`, `STATICFILES_DIRS = [BASE_DIR/static]`
- Media: `MEDIA_ROOT = BASE_DIR/media`
- Cache: LocMemCache (for rate limiting)
- Custom context processors: `language.context.user_context` / `site_context` — injects user role and AMap key
- Custom auth bridge: `language.middleware.SessionUserMiddleware` — maps the app session user to `request.user` for Django/DRF permissions

### AI
- AI: OpenAI-compatible API (default Moonshot/Kimi endpoint `https://api.moonshot.cn/v1`)
- AI views: `Dashboard/ai_views.py` — translation, visualization, chat, web search
- Chat history: `User.models.ChatHistory` — limited to 50 per user session
- Retry config: 3 retries with 1s initial delay, exponential backoff

### Authentication
- Session-based auth via `language.utils.login_required_custom` decorator
- User model uses `username` as primary key (max 20 chars)
- Roles: `admin` (0/1), `expert` (0/1), `reputation_score`
- Expert certification: `ExpertApplication` model with approval workflow
- User profile: `UserProfile` with avatar, bio, phone, location, website

### Community Features
- Post moderation queue (`ContentModeration`)
- Reply nesting (children)
- Like/Collect on posts and replies
- Follow users
- Messages (DM + system)
- Categories & Tags for posts
- Report system for content moderation

### Template Tags
- `custom_tags` (Dashboard/templatetags/) — custom template filters

### Key Dependencies
- Django 4.2, DRF, coreapi
- MySQL client (`mysqlclient`)
- OpenAI SDK (for AI features)
- Pillow (for image uploads)
- python-dotenv
- requests (for web search)
