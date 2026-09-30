<script lang="ts">
	// Apply shared styles to every route, and give every page its link preview tags.
	import './layout.css';
	import { page } from '$app/state';

	let { children } = $props();

	const SITE = 'https://mise.art';
	const TAGLINE = 'every feeling has a world.';

	/** A share link carries its feeling in page data. The preview then names the feeling. */
	const feeling = $derived(typeof page.data.text === 'string' ? page.data.text.trim() : '');
	const title = $derived(feeling ? `“${feeling}”` : 'mise');
	const url = $derived(new URL(page.url.pathname, SITE).href);
	/**
	 * A share link shows its feeling in the feeling's own colors, also for a path through its worlds.
	 * Every other page shows the brand card.
	 */
	const code = $derived(
		feeling && typeof page.data.feeling === 'string' ? page.data.feeling : undefined
	);
	const image = $derived(code ? `${SITE}/og/${code}.png` : `${SITE}/brand/og.png`);
	const alt = $derived(
		code ? `${title} on mise: every feeling has a world.` : 'mise: every feeling has a world.'
	);
	/** The mood page sets its own tab icon from the palette, so it must not get a second one. */
	const ownIcon = $derived(page.route.id === '/[[code=code]]');
</script>

<svelte:head>
	<link rel="canonical" href={url} />
	{#if !ownIcon}
		<link rel="icon" href="/favicon.svg" type="image/svg+xml" />
	{/if}
	<meta property="og:title" content={title} />
	<meta property="og:description" content={TAGLINE} />
	<meta property="og:url" content={url} />
	<meta property="og:image" content={image} />
	<meta property="og:image:type" content="image/png" />
	<meta property="og:image:width" content="1200" />
	<meta property="og:image:height" content="630" />
	<meta property="og:image:alt" content={alt} />
	<meta name="twitter:title" content={title} />
	<meta name="twitter:description" content={TAGLINE} />
	<meta name="twitter:image" content={image} />
	<meta name="twitter:image:alt" content={alt} />
</svelte:head>

{@render children()}
