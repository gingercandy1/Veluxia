SOURCES += $$files(../src/**/*.py, true)

FORMS += ../ui/*.ui
FORMS += $$files(src/**/*.ui, true)

TRANSLATIONS += \
    translations/app_zh_CN.ts \
    translations/app_en_US.ts \
    translations/app_ja_JP.ts \
    translations/app_ko_KR.ts \
    translations/app_ru_RU.ts