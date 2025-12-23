{% extends "!autosummary/class.rst" %}

{% block methods %}
{% if methods %}
{{ super() }}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
{{ super() }}
{% endif %}
{% endblock %}