"""
Post-processing : Command Validator.

Quatre niveaux de protection :
  1. Longueur minimale — réponse trop courte = fragment LLM → bloqué
  2. Mélange de technologies — SSH + IPsec = hallucination → bloqué
  3. Whitelist par technologie — pfSense GUI only, SSH, strongSwan, FreeBSD, Linux
  4. Fallback contextuel — commande absente du contexte RAG → bloqué
"""

import re
from loguru import logger

# ── Whitelists par technologie ────────────────────────────────────────────────

# pfSense = GUI uniquement, aucune commande CLI
_PFSENSE_WHITELIST: frozenset[str] = frozenset()

# SSH — commandes client/serveur + outils de diagnostic
_SSH_WHITELIST: frozenset[str] = frozenset({
    "ssh", "scp", "sftp", "sshd",
    "ssh-keygen", "ssh-copy-id", "ssh-add", "ssh-agent",
    "journalctl",                   # diagnostic sshd via journald
    "systemctl",                    # start/stop sshd
    "sudo",                         # édition fichiers system
    "grep", "cat", "tail", "less",  # lecture des logs
    "vi", "nano", "vim",            # édition sshd_config
    "awk", "sed", "head",           # traitement logs
    # Fichiers de logs SSH courants (noms extraits des chemins)
    "syslog", "auth.log", "secure", "messages",
})

# strongSwan — token racine "swanctl" couvre toutes les sous-commandes
_STRONGSWAN_WHITELIST: frozenset[str] = frozenset({
    "swanctl",
})

# FreeBSD pf — token racine "pfctl"
_FREEBSD_PF_WHITELIST: frozenset[str] = frozenset({
    "pfctl",
})

# Linux — administration système & réseau
_LINUX_WHITELIST: frozenset[str] = frozenset({
    "ip", "iptables", "ip6tables", "nft",
    "systemctl", "service", "journalctl",
    "apt", "apt-get", "yum", "dnf", "pacman",
    "ssh", "scp", "sshd", "ssh-keygen", "ssh-copy-id",
    "curl", "wget",
    "cat", "grep", "sed", "awk", "tail", "head", "less", "more",
    "chmod", "chown", "ls", "mkdir", "rm", "cp", "mv", "ln",
    "ping", "traceroute", "netstat", "ss", "tcpdump", "nmap",
    "firewall-cmd", "ufw",
    "adduser", "useradd", "usermod", "passwd", "groupadd",
    "wazuh-agent", "wazuh-control", "wazuh-agentd", "ossec-control",
    # Noms de fichiers de config courants (vérifiés sans le chemin complet)
    "wazuh.conf", "ossec.conf", "sshd_config", "ssh_config", "sudoers",
    "zabbix_agent2.conf", "zabbix_agentd.conf", "zabbix_server.conf",
    "mount", "umount", "df", "du", "lsblk",
    "tar", "gzip", "zip", "unzip",
    "python", "python3", "bash", "sh",
    "nano", "vim", "vi",
    "echo", "printf", "tee",
    "sudo", "su",
    "find", "locate", "which",
    "ps", "top", "htop", "kill", "pkill",
})

_TECH_WHITELISTS: dict[str, frozenset[str]] = {
    "pfsense":    _PFSENSE_WHITELIST,
    "ssh":        _SSH_WHITELIST,
    "strongswan": _STRONGSWAN_WHITELIST,
    "freebsd":    _FREEBSD_PF_WHITELIST,
    "linux":      _LINUX_WHITELIST,
}

# ── Détection de technologie ──────────────────────────────────────────────────
# Ordre de priorité (premier match gagne)

_TECH_KEYWORDS: list[tuple[str, frozenset[str]]] = [
    ("pfsense",    frozenset({"pfsense", "netgate"})),
    ("strongswan", frozenset({"strongswan", "swanctl", "charon"})),
    ("freebsd",    frozenset({"freebsd", "pfctl", "openbsd"})),
    # SSH avant Linux : question SSH → whitelist SSH (inclut journalctl)
    ("ssh",        frozenset({
        "sshd_config", "sshd", "authorized_keys", "known_hosts",
        "openssh", "/etc/ssh",
    })),
    ("linux",      frozenset({
        "linux", "ubuntu", "debian", "centos", "rhel", "fedora",
        "journalctl", "systemctl", "sudoers",
        "tcpdump", "iptables", "journald", "systemd", "bash",
        "/etc/", "/var/log/", "/usr/bin/",
    })),
]


def _detect_tech(text: str) -> str | None:
    """Retourne la technologie principale détectée dans le texte (question + contexte)."""
    t = text.lower()
    for tech, keywords in _TECH_KEYWORDS:
        if any(kw in t for kw in keywords):
            return tech
    return None


def _detect_tech_priority(question: str, context: str) -> str | None:
    """
    Détecte la technologie en donnant la PRIORITÉ à la question.
    Évite qu'un contexte pfSense override une question explicitement Linux.
    """
    # 1. La question seule (priorité maximale)
    tech_from_q = _detect_tech(question)
    if tech_from_q:
        return tech_from_q
    # 2. Contexte en fallback
    return _detect_tech(context)


# ── Détection du mélange de technologies ─────────────────────────────────────

_CMD_FAMILY: dict[str, str] = {
    # IPsec / strongSwan
    "swanctl": "ipsec",
    "charon":  "ipsec",
    "ipsec":   "ipsec",
    # Linux
    "iptables":    "linux",
    "ip6tables":   "linux",
    "ip":          "linux",
    "nft":         "linux",
    "systemctl":   "linux",
    "apt":         "linux",
    "apt-get":     "linux",
    "yum":         "linux",
    "dnf":         "linux",
    "firewall-cmd":"linux",
    "ufw":         "linux",
    # BSD / pfSense
    "pfctl":   "bsd",
    "kldload": "bsd",
    "sysctl":  "bsd",
    # SSH server config (pas les commandes ssh courantes)
    "sshd":        "ssh_srv",
}

_INCOMPATIBLE_PAIRS: list[frozenset[str]] = [
    frozenset({"ipsec", "ssh_srv"}),  # swanctl + sshd dans la même réponse
    frozenset({"bsd", "linux"}),      # pfctl + iptables → OS différents
]


def _detect_tech_mixing(tokens: set[str]) -> tuple[bool, list[str]]:
    families = {_CMD_FAMILY[t] for t in tokens if t in _CMD_FAMILY}
    for pair in _INCOMPATIBLE_PAIRS:
        if pair.issubset(families):
            return True, sorted(families)
    return False, []


# ── Messages de refus ─────────────────────────────────────────────────────────

_ABSENT_MSG = (
    "**Information absente du contexte.**\n\n"
    "La documentation disponible ne contient pas les informations nécessaires "
    "pour répondre à cette question avec des commandes vérifiées. "
    "Aucune commande ne peut être fournie sans risque d'erreur."
)

_PFSENSE_MSG = (
    "**pfSense — Interface graphique uniquement.**\n\n"
    "pfSense se configure exclusivement via son interface web (GUI). "
    "Aucune commande CLI n'est applicable ici. "
    "La réponse générée contenait des commandes CLI qui ne correspondent "
    "pas à pfSense et ont été bloquées automatiquement."
)

_MIXING_MSG = (
    "**Incohérence technologique détectée.**\n\n"
    "La réponse générée mélange des commandes de technologies incompatibles "
    "(ex : commandes IPsec et configuration SSH, ou commandes Linux et BSD). "
    "Aucune réponse fiable ne peut être fournie. "
    "Reformulez votre question en ciblant une seule technologie."
)

_SHORT_MSG = (
    "**Réponse insuffisante.**\n\n"
    "Le modèle n'a pas produit une réponse complète pour cette question. "
    "Veuillez reformuler ou réessayer."
)

# ── Longueur minimale ─────────────────────────────────────────────────────────

_MIN_RESPONSE_CHARS = 60

# Marqueurs indiquant que la réponse est déjà un message du validator
_VALIDATOR_MARKERS = (
    "absente du contexte",
    "GUI uniquement",
    "Incohérence technologique",
    "Réponse insuffisante",
)

# ── Faux positifs CamelCase à ignorer ────────────────────────────────────────

_IGNORE_TOKENS: frozenset[str] = frozenset({
    # Mots anglais courants terminant en -al/-ical
    "protocol", "control", "article", "electrical", "optical",
    "technical", "identical", "critical", "vertical", "typical",
    "central", "practical", "chemical", "physical", "logical",
    "medical", "classical", "musical", "radical", "neutral",
    # ── Directives sshd_config (CamelCase = faux positifs) ──
    "syslogfacility", "authenticationmethods", "permittunnel",
    "allowagentforwarding", "allowtcpforwarding", "gatewayports",
    "permituserenvironment", "strictmodes", "pubkeyauthentication",
    "passwordauthentication", "challengeresponseauthentication",
    "kerberosauthentication", "hostbasedauthentication",
    "ignoreuserknownhosts", "ignorerhosts", "usepam",
    "printmotd", "printlastlog", "tcpkeepalive", "xauthlocation",
    "subsystem", "maxauthtries", "maxsessions",
    "logingracetime", "logingracelogin", "listenaddress",
    "addressfamily", "hostkey", "rekeylimit",
    "clientaliveinterval", "clientalivecountmax",
    # ── Directives réseau / config génériques ──
    "nameserver", "searchdomain", "localhost", "loopback",
    "defaultroute", "netmask", "broadcast",
})

# Un token valide doit contenir au moins un caractère alphanumérique
_RE_ALNUM = re.compile(r"[a-zA-Z0-9]")

# ── Patterns d'extraction ─────────────────────────────────────────────────────

_RE_CODE_BLOCK  = re.compile(r"```(?:[a-z]*\n?)?([\s\S]+?)```")
_RE_INLINE_CODE = re.compile(r"`([^`\n]{2,80})`")
_RE_CMD_LINE    = re.compile(r"(?m)^[$#>]\s*(.+)$")
_RE_CTL_TOKEN   = re.compile(r"\b(\w+ctl)\b", re.IGNORECASE)
_RE_PATH_TOKEN  = re.compile(r"(/(?:bin|sbin|usr|etc|var|opt)[^\s`\"'\\]+)")
_RE_SUDO        = re.compile(r"\bsudo\s+(\S+)")
_RE_CAMEL_CMD   = re.compile(
    r"\b(?:[A-Z]{2,}[a-z]\w{3,}|[A-Z][a-z]+[A-Z]\w{3,})\b"
)
_RE_ENABLE_CMD  = re.compile(r"\benable\w{4,}\b", re.IGNORECASE)


def _extract_command_tokens(text: str) -> set[str]:
    tokens: set[str] = set()

    # 1. Blocs de code ```...```
    for m in _RE_CODE_BLOCK.finditer(text):
        for line in m.group(1).strip().splitlines():
            line = line.strip().lstrip("$#> ")
            if not line:
                continue
            first = line.split()[0].lower()
            if len(first) > 2 and first not in _IGNORE_TOKENS and _RE_ALNUM.search(first):
                tokens.add(first)

    # 2. Code inline `commande arg`
    for m in _RE_INLINE_CODE.finditer(text):
        tok = m.group(1).strip()
        first = tok.split()[0].lower()
        if len(first) > 2 and first not in _IGNORE_TOKENS and _RE_ALNUM.search(first):
            tokens.add(first)

    # 3. Lignes shell ($, #, >)
    for m in _RE_CMD_LINE.finditer(text):
        line = m.group(1).strip()
        if line:
            first = line.split()[0].lower()
            if len(first) > 2 and _RE_ALNUM.search(first):
                tokens.add(first)

    # 4. Tokens *ctl (swanctl, systemctl, pfctl…)
    for m in _RE_CTL_TOKEN.finditer(text):
        tok = m.group(1).lower()
        if tok not in _IGNORE_TOKENS:
            tokens.add(tok)

    # 5. Chemins Unix absolus (/etc/, /bin/…) — strip ponctuation finale
    for m in _RE_PATH_TOKEN.finditer(text):
        tok = m.group(1).rstrip(".,;:!?'\"`")
        tokens.add(tok)

    # 6. sudo <commande>
    for m in _RE_SUDO.finditer(text):
        tokens.add(m.group(1).lower())

    # 7. CamelCase technique (SSHdKeyOnly…) — directives sshd_config filtrées par _IGNORE_TOKENS
    for m in _RE_CAMEL_CMD.finditer(text):
        tok = m.group(0).lower()
        if tok not in _IGNORE_TOKENS:
            tokens.add(tok)

    # 8. enable* composés (enablesshd, enablefirewall…)
    for m in _RE_ENABLE_CMD.finditer(text):
        tokens.add(m.group(0).lower())

    return tokens


# ── Niveau 0 : longueur minimale ─────────────────────────────────────────────

def validate_length(response: str) -> str:
    """
    Bloque les réponses trop courtes (fragment de token, identifiant isolé).
    Ne s'applique pas aux messages du validator lui-même.
    """
    clean = response.strip()
    if any(marker in clean for marker in _VALIDATOR_MARKERS):
        return response  # déjà un message de refus du validator
    if len(clean) < _MIN_RESPONSE_CHARS:
        logger.warning(
            f"[CommandValidator] Réponse trop courte ({len(clean)} chars) — bloquée."
        )
        return _SHORT_MSG
    return response


# ── Niveaux 1-3 : validation des commandes ───────────────────────────────────

def validate_commands(response: str, context: str, question: str = "") -> str:
    """
    Niveau 1 — Mélange de technologies (ipsec+ssh_srv, bsd+linux)
    Niveau 2 — Whitelist par technologie :
                pfSense → GUI only (0 commande)
                SSH     → ssh, sshd, journalctl, systemctl…
                strongSwan → swanctl uniquement
                FreeBSD → pfctl uniquement
                Linux   → liste étendue
    Niveau 3 — Présence dans le contexte RAG (fallback)
    """
    tokens = _extract_command_tokens(response)
    if not tokens:
        return response

    # ── Niveau 1 : mélange de technologies ───────────────────────────────────
    mixing, families = _detect_tech_mixing(tokens)
    if mixing:
        logger.warning(
            f"[CommandValidator] Mélange technologique — familles : {families}, "
            f"tokens : {sorted(tokens)} — réponse bloquée."
        )
        return _MIXING_MSG

    # ── Niveau 2 : whitelist par technologie ─────────────────────────────────
    tech = _detect_tech_priority(question, context)

    if tech == "pfsense":
        logger.warning(
            f"[CommandValidator] pfSense (GUI only) — {len(tokens)} commande(s) "
            f"CLI bloquée(s) : {sorted(tokens)}"
        )
        return _PFSENSE_MSG

    if tech in _TECH_WHITELISTS:
        whitelist = _TECH_WHITELISTS[tech]
        ctx_lower = context.lower()
        out_of_whitelist = [
            t for t in sorted(tokens)
            if t not in whitelist
            and t not in ctx_lower
            and t.split("/")[-1] not in ctx_lower
            # Fichiers de config connus (ex: wazuh.conf dans /etc/wazuh/wazuh.conf)
            and t.split("/")[-1] not in whitelist
        ]
        if out_of_whitelist:
            logger.warning(
                f"[CommandValidator] Commandes hors whitelist '{tech}' : "
                f"{out_of_whitelist} — réponse bloquée."
            )
            return _ABSENT_MSG
        # Level 2 validé → pas besoin de Level 3
        logger.debug(
            f"[CommandValidator] {len(tokens)} commande(s) validée(s) "
            f"[tech={tech}] : {sorted(tokens)}"
        )
        return response

    # ── Niveau 3 : fallback contextuel (technologie non reconnue) ────────────
    ctx_lower = context.lower()
    invalid = [
        tok for tok in sorted(tokens)
        if tok not in ctx_lower
        and tok.split("/")[-1] not in ctx_lower
    ]
    if invalid:
        logger.warning(
            f"[CommandValidator] Commandes absentes du contexte RAG : {invalid} "
            f"— réponse bloquée."
        )
        return _ABSENT_MSG

    logger.debug(
        f"[CommandValidator] {len(tokens)} commande(s) validée(s) "
        f"[tech=inconnue] : {sorted(tokens)}"
    )
    return response
