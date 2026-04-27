import argostranslate


print(argostranslate.package.get_installed_packages())
print([l.code for l in argostranslate.translate.get_installed_languages()])