<script lang="ts">
	import { resolve } from '$app/paths';
</script>

<svelte:head>
	<title>attribution — mise</title>
</svelte:head>

<main class="page">
	<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
	<a class="back" href={resolve('/')}>back to mise</a>

	<h1>attribution</h1>
	<p class="intro">mise uses data and images from these sources.</p>

	<ul class="sources">
		<li>
			<p class="head">
				<img class="logo" src="/attribution/tmdb.svg" alt="" width="92" height="12" />
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://www.themoviedb.org/" target="_blank" rel="noopener">TMDB</a>
				— film titles, posters, and cast and crew credits.
			</p>
			<p class="notice">
				This application uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise
				approved by TMDB.
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://hardcover.app/" target="_blank" rel="noopener">Hardcover</a>
				— book titles, covers, and editions.
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://www.last.fm/" target="_blank" rel="noopener">Last.fm</a>
				— listening and tag data for songs.
			</p>
			<p class="notice">Data from Last.fm.</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://www.deezer.com/" target="_blank" rel="noopener">Deezer</a>
				— song previews, cover art, and links to full tracks.
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://musicbrainz.org/" target="_blank" rel="noopener">MusicBrainz</a>
				and
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://listenbrainz.org/" target="_blank" rel="noopener">ListenBrainz</a>
				— matching songs and artists across sources. Public domain (CC0).
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://www.metmuseum.org/" target="_blank" rel="noopener"
					>The Metropolitan Museum of Art</a
				>
				— public-domain artwork images and data. Public domain (CC0).
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://www.artic.edu/" target="_blank" rel="noopener">Art Institute of Chicago</a>
				— public-domain artwork images and data. Public domain (CC0).
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://www.clevelandart.org/" target="_blank" rel="noopener"
					>Cleveland Museum of Art</a
				>
				— public-domain artwork images and data. Public domain (CC0).
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://commons.wikimedia.org/" target="_blank" rel="noopener">Wikimedia Commons</a
				>
				— public-domain scans of artwork.
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://openlibrary.org/" target="_blank" rel="noopener">Open Library</a>
				— book cover images, as a fallback.
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://poetrydb.org/" target="_blank" rel="noopener">PoetryDB</a>
				— public-domain poems.
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://arxiv.org/abs/1804.04128" target="_blank" rel="noopener"
					>Text2Colors (Bahng et al.)</a
				>
				— training data for the color-palette model.
			</p>
			<p class="notice">
				Hyojin Bahng, Seungjoo Yoo, Wonwoong Cho, David Keetae Park, Ziming Wu, Xiaojuan Ma, and
				Jaegul Choo, "Coloring with Words: Guiding Image Colorization through Text-based Palette
				Generation," ECCV 2018.
			</p>
		</li>

		<li>
			<p class="head">
				<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- external link -->
				<a href="https://fonts.google.com/" target="_blank" rel="noopener">Google Fonts</a>
				— typefaces, loaded to match each mood.
			</p>
		</li>
	</ul>
</main>

<style>
	.page {
		box-sizing: border-box;
		width: min(100% - 2 * max(2.5vw, 12px), 40rem);
		margin: 0 auto;
		padding: clamp(4rem, 10vh, 6rem) 0 4rem;
	}

	.back {
		display: inline-flex;
		align-items: center;
		min-height: 44px;
		margin: -0.6rem 0 1.5rem -0.1rem;
		padding: 0.6rem 0.1rem;
		color: color-mix(in oklab, var(--ink) 55%, transparent);
		font-size: 0.85rem;
		font-style: normal;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.12em;
		text-decoration: none;
		transition: color 300ms var(--ease);
	}

	.back:hover,
	.back:focus-visible {
		color: var(--ink);
	}

	.back:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: -2px;
	}

	h1 {
		margin: 0 0 0.5rem;
		font-size: clamp(1.8rem, 3vw + 0.6rem, 2.6rem);
		font-weight: 400;
		font-variant-caps: all-small-caps;
		letter-spacing: 0.02em;
		color: var(--ink);
	}

	.intro {
		margin: 0 0 2.5rem;
		color: color-mix(in oklab, var(--ink) 75%, transparent);
		font-size: 1.05rem;
	}

	.sources {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.sources li {
		padding-top: 1.5rem;
		border-top: 1px solid color-mix(in oklab, var(--ink) 12%, transparent);
	}

	.sources li:first-child {
		padding-top: 0;
		border-top: 0;
	}

	.head {
		margin: 0;
		color: var(--ink);
		font-size: 1.02rem;
		line-height: 1.55;
	}

	.head a {
		color: var(--ink);
		text-decoration: none;
		border-bottom: 1px solid color-mix(in oklab, var(--ink) 40%, transparent);
		transition: border-color 300ms var(--ease);
	}

	.head a:hover,
	.head a:focus-visible {
		border-color: var(--ink);
	}

	.logo {
		display: inline-block;
		height: 0.75em;
		width: auto;
		margin-right: 0.35rem;
		vertical-align: baseline;
		opacity: 0.85;
	}

	.notice {
		margin: 0.4rem 0 0;
		color: color-mix(in oklab, var(--ink) 55%, transparent);
		font-size: 0.9rem;
		line-height: 1.5;
	}
</style>
